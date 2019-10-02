#usr/bin/env python
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt                             
import os, numpy as np, time, cPickle as pickle
import sim, pickleTraj
from spline2gaussians_leastsquares import GaussianBasisLSQ # This also requires spline to be imported
import re         

print sim

#os.system('rm -rf pylib\n')
#os.system('rm *pyc\n')
#os.system('rm tmp*\n')
#os.system('rm *pickle\n')

Traj_List = TrajList_DUMMY
NMol_List = NMol_DUMMY
DOP = DOP_DUMMY
MappingRatio = CGMap_DUMMY
Pressure_List = Pressure_List_DUMMY 
StageCoefs = StageCoefs_DUMMY  #HERE may need to be more gradual for stability
SrelName = "CG_run"
UseLammps = UseLammps_DUMMY 
UseOMM = UseOMM_DUMMY
UseSim = UseSim_DUMMY


# Default Simulation Package Settings
sim.export.lammps.InnerCutoff = 0.000000001
sim.export.lammps.NPairPotentialBins = 1000000
sim.export.lammps.LammpsExec = 'lmp_omp'
sim.export.lammps.UseLangevin = True
sim.export.lammps.OMP_NumThread = Threads_DUMMY
sim.export.lammps.TableInterpolationStyle = 'lookup' # More robust than spline for highly CG-ed systems
sim.srel.optimizetrajlammps.LammpsDelTempFiles = False
sim.srel.optimizetrajlammps.UseLangevin = True

sim.export.omm.platformName = 'OpenCL' # or 'OpenCL' or 'GPU' or 'CUDA'
sim.export.omm.device = -1 #-1 is default, let openmm choose its own platform.
sim.export.omm.NPairPotentialKnots = 500 #number of points used to spline-interpolate the potential
sim.export.omm.InnerCutoff = 0.001 #0.001 is default. Note that a small value is not necessary, like in the lammps export, because the omm export interpolates to zero
sim.srel.optimizetrajomm.OpenMMStepsMin = 0 #number of steps to minimize structure, 0 is default
sim.srel.optimizetrajomm.OpenMMDelTempFiles = False #False is Default
sim.export.omm.UseTabulated = True

#Srel Settings
sim.srel.optimizetraj.CGGradTol = 1E-3
sim.srel.optimizetraj.CGAbsTol = 1E-7
sim.srel.optimizetraj.CGFracTol = 1E-7

print('CGGradTol: {}'.format(sim.srel.optimizetraj.CGGradTol))
print('CGAbsTol: {}'.format(sim.srel.optimizetraj.CGAbsTol))
print('CGFracTol: {}'.format(sim.srel.optimizetraj.CGFracTol))

#External Potential Settings
Ext = {"UConst": UConst_DUMMY, "NPeriods": NPeriods_DUMMY, "PlaneAxis": PlaneAxis_DUMMY, "PlaneLoc": PlaneLoc_DUMMY}
if Ext["UConst"] > 0:
    print("Using external sinusoid with UConst {}".format(Ext["UConst"]))
    UseExternal = True
else:
    UseExternal = False


# MD Iterations
StepsEquil 			= StepsEquil_DUMMY
StepsProd 			= StepsProd_DUMMY
StepsStride 		= StepsStride_DUMMY
ScaleRuns 			= ScaleRuns_DUMMY
RunStepScaleList 	= RunStepScaleList_DUMMY
GaussMethod 		= GaussMethod_DUMMY


# Force-Field Settings
Cut             = Cut_DUMMY
FixBondDist0    = True
BondFConst      = BondFConst_DUMMY
PBondDist0      = 0. # For zero centered bonds set to 0.
IncludeBondedAtoms  = IncludeBondedAtoms_DUMMY
UseLocalDensity     = False                                                                      
CoordMin        = 0    
CoordMax        = 350
LDKnots         = 10
NumberGaussians = NumberGaussians_DUMMY
RunSpline       = RunSpline_DUMMY
NSplineKnots    = NSplineKnots_DUMMY
SplineKnots     = SplineKnots_DUMMY

# Spline options
#   Option1 = Constant slope 
#   Option2 = Constant slope, then turn-off
#   Option3 = Slope unconstrained

SplineOption        = SplineOption_DUMMY 
FitSpline           = FitSpline_DUMMY # Turns on Gaussian Fit of the spline for the initial guess
SysLoadFF           = SysLoadFF_DUMMY # Use if you desire to seed a run with an already converged force-field.
force_field_file    = force_field_file_DUMMY               
UseWPenalty         = UseWPenalty_DUMMY
WriteTraj           = True
UseExpandedEnsemble = UseExpandedEnsemble_DUMMY
RunConvergedCGModel = True # Run the converged ff file at the end (calculates P and Rg statistics), 
WeightSysByMolecules = False # Option to weight E.E. systems by the number of molecules
WeightSysByMoleculeRatios = True # Option to weight E.E. systems by the ratio of number of molecules

''' Bulk of Code '''
def FreezeParameters(System_List, Pot, Parameters):
    # - Pot is the index of the potential with parameters to freeze.
    # - Parmaters is a list of parameters to freeze in Pot.
    for index, Sys in enumerate(System_List):
        for P_index, Pot in enumerate(Sys.ForceField):
            if P_index == Pot:
                Pot.FreezeSpecificParam(Parameters) 
                

def CreateForceField(Sys, Cut, UseLocalDensity, CoordMin, CoordMax, LDKnots, RunSpline, 
                        NSplineKnots, NumberGaussians, GaussMethod, SplineKnots):
    ''' Function that creates the system force-field. '''
    
    opt = None
    FFList = []
    FFGaussians = []
    AtomType = Sys.World[0][0] #since only have one atom type in the system
    
    ''' Add in potentials '''
    # Add PBond, Always assumed to be the first potential object!
    PBond = sim.potential.Bond(Sys, Filter = sim.atomselect.BondPairs,
                               Dist0 = PBondDist0, FConst = .1, Label = 'Bond')
    
    PBond.Param.Dist0.Min = 0.
    FFList.extend([PBond])
    
    if GaussMethod in {4,5,6,7,8,9,10}:
        PBond.Param.FConst = BondFConst
    
    if IncludeBondedAtoms:
        Filter = sim.atomselect.Pairs
    else:
        Filter = sim.atomselect.NonbondPairs12
    ''' Add Splines '''
    if RunSpline:
        PSpline = sim.potential.PairSpline(Sys, Filter = Filter, Cut = Cut,
                                           NKnot = NSplineKnots, Label = 'Spline', 
                                           NonbondEneSlopeInit = "0.25kTperA", BondEneSlope = "0.25kTperA")
        if FitSpline:
            Max = 4.
            decay = 0.1
            ArgVals = np.linspace(0,Cut,100)
            ArgValsSq = np.multiply(ArgVals,ArgVals)
            EneVals = Max*np.exp(-1.*decay*ArgValsSq)
            PSpline.FitSpline(ArgVals, EneVals, Weights = None, MaxKnot = None, MinKnot = None)
        
        PSpline.KnotMinHistFrac = 0.005
        if SplineOption == 'Option3':
            PSpline.EneSlopeInner = None                             
        
        FFList.append(PSpline)
    
    else:
        # lj gauss
        GaussPot_List = []
        
        if GaussMethod in {4,5,6,7,8,9,10}: # Fitting Gaussians to spline
            opt = GaussianBasisLSQ(knots=SplineKnots, rcut=Cut, rcutinner=0., ng=10, nostage=False, N=2000, BoundSetting='Option1', U_max_2_consider=2.5, 
                        SlopeCut=-1., ShowFigures=False, SaveToFile=True, SaveFileName = 'GaussianLSQFitting',
                        weight_rssq = True, Cut_Length_Scale=1.,TailCorrection=False, TailWeight=1E6)
            
            print('\nOptimal number of Gaussians are {}\n'.format(opt[0]))
            print('Optimal parameters: \n')
            print('\n{}\n'.format(opt[1]))
            
            NumberGaussians = opt[0]
            
        for g in range(NumberGaussians):
            temp_Label = 'LJGauss{}'.format(g)
            temp_Gauss = sim.potential.LJGaussian(Sys, Filter = Filter, Cut = Cut,
                                 Sigma = 1.0, Epsilon = 0.0, B = 0., Kappa = 0.15,
                                 Dist0 = 0.0, Label = temp_Label)
            
            temp_Gauss.MinHistFrac = 0.01
            temp_Gauss.Param.Sigma.Fixed = True
            temp_Gauss.Param.Epsilon.Fixed = True
            temp_Gauss.Param.Dist0.Fixed = True
            
            if g == 0: # initialize the first Guassian to be repulsive
                temp_Gauss.Param.B = 5.
                temp_Gauss.Param.B.min = 0.
                
            if GaussMethod in {4,5,6,7,8,9,10}: # Set the initial guesses on the parameters
                temp_Gauss.Param.B = opt[1][g*2]
                temp_Gauss.Param.Kappa = opt[1][g*2+1]
                
            GaussPot_List.append(temp_Gauss)
        
        FFGaussians.extend(GaussPot_List)
         
    # LocalDensity Function not currently tested
    if UseLocalDensity: 
        CoordMin = CoordMin # coordination number minimum
        CoordMax = CoordMax  # coordination number maximum
        NKnot = LDKnots
        Cut = Cut
        FilterAA = sim.atomselect.PolyFilter([AtomType, AtomType], Ordered = True)
        print ('Filter AA {}'.format(FilterAA))
        PLD_AA = sim.potential.LocalDensity(Sys, Cut = Cut, InnerCut = 0.75*Cut,
                                     Knots = None, NKnot = NKnot,
                                     RhoMin = CoordMin, RhoMax = CoordMax, 
                                     Label = "LD", Filter = FilterAA)
        # add reqd. potentials to the forcefield
        FFList.extend([PLD_AA])
    
   #--- External Potential---
    if UseExternal:
        """applies external field to just species included in FilterExt"""
        FilterExt = sim.atomselect.PolyFilter([AtomType])
        ExtPot = sim.potential.ExternalSinusoid(Sys, Filter=FilterExt, UConst=Ext["UConst"], NPeriods=Ext["NPeriods"], PlaneAxis=Ext["PlaneAxis"], PlaneLoc=Ext["PlaneLoc"], Label="ExtSin")
	FFList.append(ExtPot)

    return FFList, FFGaussians, NumberGaussians, opt

def CreateSystem(Name, BoxL, NumberMolecules, NumberMonomers, Cut, UseLocalDensity, CoordMin, CoordMax,
                    LDKnots, RunSpline, NSplineKnots, NumberGaussians):
    ''' Function that creates the system objects and returns them and a force-field object list. '''
    
    FFList = []
    SysName = Name
    NMol = NumberMolecules
    NMon = NumberMonomers
    print ('NumberMonomers')
    print (NumberMonomers)
    TempSet = 1.0

    AtomType = sim.chem.AtomType("A", Mass = 1., Charge = 0.0)
    MolType = sim.chem.MolType("M", [AtomType]*NMon)
    World = sim.chem.World([MolType], Dim = 3, Units = sim.units.DimensionlessUnits)
   
    #create bonds between monomer pairs
    for bond_index in range(0, NMon-1):
        MolType.Bond(bond_index, bond_index+1)

    # make system    
    Sys = sim.system.System(World, Name = SysName)
    for i in range(NMol):
        Sys += MolType.New()

    ''' Add in potentials '''
    
    FFList, FFGaussians, NumberGaussians, opt = CreateForceField(Sys, Cut, UseLocalDensity, CoordMin, CoordMax, LDKnots, 
                                                            RunSpline, NSplineKnots,
                                                            NumberGaussians, GaussMethod, SplineKnots) 
                                
    Sys.ForceField.extend(FFList)
    Sys.ForceField.extend(FFGaussians)
    

    #set up the histograms
    for P in Sys.ForceField:
        P.Arg.SetupHist(NBin = 8000, ReportNBin = 200)

    # lock and load
    Sys.Load()

    Sys.BoxL = BoxL

    #initial positions and velocities
    sim.system.positions.CubicLattice(Sys)
    sim.system.velocities.Canonical(Sys, Temp = 1.0)
    Sys.TempSet = 1.0

    #configure integrator
    Int = Sys.Int

    Int.Method = Int.Methods.VVIntegrate        
    Int.Method.Thermostat = Int.Method.ThermostatLangevin
    Int.Method.TimeStep = TimeStep_DUMMY # note: reduced units
    Int.Method.LangevinGamma = 1/(100*Int.Method.TimeStep)
    Sys.TempSet = TempSet
    
    return Sys, [FFList, FFGaussians], NumberGaussians, opt

    
''' ************************************************************** '''
''' ********** Generate Systems & Optimizers for Srel ************ '''
''' ************************************************************** '''

SysList = []
OptList = []
NumAtomsList = [] # Might be useful for weighting expanded ensemble runs
SysFFList = [] # A list of force field list for each system!

for index, NMol in enumerate(NMol_List):
    ''' Setup Systems '''
    # ALWAYS TREATS EVERYTHING LIKE MULTITRAJ
    CGDOP = DOP/MappingRatio
    Name = ('Sys_NMol_'+str(NMol))
    
    # Load in trajectories and perform mapping.
    # N.S. TODO: Make compatible with pickleTraj.py script
    Traj = Traj_List[index]
    if MappingRatio != 1:
        OutTraj = Traj.split('.lammpstrj')[0] + '_mapped.lammpstrj'
        Map = sim.atommap.PosMap()
        monomer_count = 0
        for chain in range(NMol): # loop over chains
            for i in range(CGDOP):
                Atoms1 = range(MappingRatio*i+monomer_count, MappingRatio*(i+1)+monomer_count)
                Atom2 = i + monomer_count/MappingRatio
                this_Map = sim.atommap.AtomMap(Atoms1 = Atoms1, Atom2 = Atom2)
                Map += [this_Map]
            monomer_count += CGDOP*MappingRatio
    #Traj_Temp = sim.traj.Lammps(Traj)
    #BoxL = Traj_Temp.Init_BoxL[0]
    Traj_Temp = pickleTraj(Traj_List[index])
    BoxL = Traj_Temp.FrameData['BoxL'] 
    if MappingRatio != 1:
        Traj_Temp = sim.traj.Mapped(Traj_Temp, Map, BoxL = BoxL)
        sim.traj.base.Convert(Traj_Temp, sim.traj.LammpsWrite, FileName = OutTraj, Verbose = True)
      
    
    SysTemp, [FFList, FFGaussians], NumberGaussians, opt = CreateSystem(Name, BoxL, NMol, CGDOP, Cut, UseLocalDensity, 
                                                        CoordMin, CoordMax, LDKnots, RunSpline, NSplineKnots, NumberGaussians)
    
    SysFFList.append([FFList, FFGaussians])

    # Freeze parameters that never change
    PBond = SysTemp.ForceField[0]
    if FixBondDist0:
        PBond.Dist0.Fixed = True
        PBond.Dist0 = PBondDist0

    ''' Now setting initial system optimizations. '''
    if SysLoadFF: # option to load in trajectory to seed the optimization
        with open(force_field_file, 'r') as of: s = of.read()
        SysTemp.ForceField.SetParamString(s)                                                        
         
    # Perform atom mapping for specific system
    MapTemp = sim.atommap.PosMap()
    print SysTemp.Name
    print 'NMol: {}'.format(SysTemp.NMol)
    print 'NAtom: {}'.format(SysTemp.NAtom)
    print 'NDOF: {}'.format(SysTemp.NDOF)
    for (i, a) in enumerate(SysTemp.Atom):
        MapTemp += [sim.atommap.AtomMap(Atoms1 = i, Atom2 = a)]
    
    ''' Setup Optimizers '''
    if UseLammps:
        OptClass = sim.srel.optimizetrajlammps.OptimizeTrajLammpsClass
    elif UseOMM:
        OptClass = sim.srel.optimizetrajomm.OptimizeTrajOpenMMClass
    else:
        OptClass = sim.srel.optimizetraj.OptimizeTrajClass
    
    SysTemp.ScaleBox(BoxL) # scale the system by the box
    
    Opt_temp = OptClass(SysTemp, MapTemp, Beta = 1./SysTemp.TempSet, Traj = Traj_Temp, FilePrefix = '{}'.format(NMol),
                        SaveLoadArgData = True, TempFileDir = os.getcwd())
                        
    # Set run times for optimization objects.
    # Useful for dilute systems versus concentration systems.
    
    if ScaleRuns:
        temp_StepsEquil = StepsEquil*RunStepScaleList[index]
        temp_StepsProd  = StepsProd*RunStepScaleList[index]
    else:
        temp_StepsEquil  = StepsEquil
        temp_StepsProd   = StepsProd
    
    Opt_temp.StepsEquil  = temp_StepsEquil
    Opt_temp.StepsProd   = temp_StepsProd
    Opt_temp.StepsStride = StepsStride
    
    # NEED TO CHECK HOW THE VIRIAL IS COMPUTED BELOW, WAS ONLY USED FOR GAUSSIAN FLUID
    if UseWPenalty == True:
        Press = Pressure_List[index]
        Volume = np.prod(BoxL)
        W = SysTemp.NDOF - 3*Press*Volume
        Opt_temp.AddPenalty("Virial", W, MeasureScale = 1./SysTemp.NAtom, Coef = 1.e-80) #HERE also need to scale the measure by 1/NAtom to be comparable to Srel
        
    OptList.append(Opt_temp)
    SysList.append(SysTemp)
    NumAtomsList.append(SysTemp.NAtom)
    
''' ******************************************* '''
''' ********** Run Srel Optimization ********** '''
''' ******************************************* '''
RunOptimization = True

def RunSrelOptimization(Optimizer, OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0):
    ''' Runs Srel Optimization '''
    Optimizer.FilePrefix = (OptimizerPrefix)
    if UseWPenalty == False:
        Optimizer.RunConjugateGradient(MaxIter=MaxIter, SteepestIter=SteepestIter)
    
    elif UseWPenalty == True:
        Optimizer.RunStages(StageCoefs = StageCoefs)


if RunOptimization:
    # Just always using the OptimizeMultiTrajClass
    Weights = [1.]*len(OptList)
    
    if WeightSysByMolecules:
        Weights = []
        for NMol in NMol_List:
            Weights.append(1./NMol)
        print ('Weights for Expanded Ensemble are:')
        print (Weights)
            
    if WeightSysByMoleculeRatios:
        Weights = []
        NMolMax = np.max(NMol_List)
        for NMol in NMol_List:
            Weights.append(NMolMax/NMol)
        print ('Weights for Expanded Ensemble are:')
        print (Weights)

    Optimizer = sim.srel.OptimizeMultiTrajClass(OptList, Weights=Weights)
    
    if RunSpline:
    
        ''' Run Splone: There are 3 options currently.
        
            Option1: Run with a constant slope
            Option2: Run with a constant slope, then turnoff
            Option3: Run entirely without a constant slope
        
        '''
        if SplineOption == 'Option1' or SplineOption == 'Option3':
            OptimizerPrefix = ("{}_OptSpline_Final".format(SrelName))
        elif SplineOption == 'Option2':
            OptimizerPrefix = ("{}_OptSpline_ConstSlope".format(SrelName))
        else:
            print('No spline option recognized or defined!')
        
        # opt. 
        RunSrelOptimization(Optimizer, OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
        
        
        if SplineOption == 'Option2':
            for SysFF in SysFFList:
                PSpline = SysFF[0][1]
		print "Relaxing spline knot constraints" 
                #PSpline.EneSlopeInner = None # Turn-off the EneSlopeInner
                PSpline.RelaxKnotConstraints() #relax all spline constraints 
            OptimizerPrefix = ("{}_OptSpline_Final".format(SrelName))
            
            # opt. 
            RunSrelOptimization(Optimizer, OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)

    else: # Run Gaussians optimization in stages
    # N.S. TODO: at somepoint could be useful to make this more robust using list so that an arbitrary number
    #                  of Gaussians can be added without manually adding in all the different stages/parameter fixing.

        ''' Run Guassians: There are 3 options currently.
        
            Option1: Run all together
            Option2: Stage one at a time; Run all together after individual
            Option3: Run First one, then allow prior and new one to float together; Run all together after individual
            Option4: Fit Gaussian to the converged spline then relax with relative entropy
            Option5: Fit Gaussians to spline and Stage one Gaussian potential at a time; Run all together after individual
            Option6: Fit Gaussians to Spline, then run all B coefficients together, followed by all kappa's 
            Option7: Fit Gaussians to Spline, then run all parameters together, but parameters are bounded within a fraction
            Option8: A mix of 6 and 7, where all B's then all Kappa's are minimized, then all parameters are varied together with constraints
            Option9: Opt all B's unconstrained (w/ Bond), then opt. Kappa and B's, but constrained
            Option10: all B's then all Kappa's are minimized, all parameters not optimized together
        '''
        
        # Option1
        #*******************************************************************************************#
        if GaussMethod == 1:
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                
                if UseLocalDensity:
                    PLD = FFList[1]
                
                for g, PGauss in enumerate(FFGaussians):
                    PGauss.FreezeSpecificParam([0,1,4]) # Freeze LJ and Gauss offsets
           	    PGauss.MinHistFrac = 0.01 
            # opt. 
            OptimizerPrefix = ("{}_OptGaussAll_Final".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
            
        #*******************************************************************************************#    
        # End Option1
        
        # Option2
        #*******************************************************************************************#
        if GaussMethod == 2: 
            #Set Kappa Constraints, s.t. kappa's for Gaussians above 1st are always longer than the first 
            Constrain_Kappa = False
            G1_Kappa_Min = None
            
            for gID in range(NumberGaussians):
                for SysFF in SysFFList: #Loop through all system FFs
                    FFList = SysFF[0] # Contains Bond and/or Splines
                    FFGaussians = SysFF[1] # Constains Gaussians
                    
                    if UseLocalDensity:
                        PLD = FFList[1]
                    
                    for index, PGauss in enumerate(FFGaussians):
                        PGauss.FreezeSpecificParam([0,1,4]) # Always Freeze LJ and Gauss offsets
                        PGauss.MinHistFrac = 0.01
                        
                        if index == 0 and gID > 0 and Constrain_Kappa == True:
                            # Get the minimum kappa
                            G1_Kappa_Min = PGauss.Kappa.Min                            
                        
                        if gID == index: # unfix parameters
                            PGauss.B.Fixed = False
                            PGauss.Kappa.Fixed = False
                            if Constrain_Kappa and G1_Kappa_Min is not None:
                                PGauss.Kappa.Min = G1_Kappa_Min
                                print('PGauss {} Kappa Minimum: {}'.format(index,G1_Kappa_Min))
                        else:
                            PGauss.B.Fixed = True
                            PGauss.Kappa.Fixed = True
                        
                # opt. 
                OptimizerPrefix = ("{}_OptGauss{}".format(SrelName,gID))
                RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
            
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                    
                for index, PGauss in enumerate(FFGaussians):
                    PGauss.B.Fixed = False
                    PGauss.Kappa.Fixed = False
            
            # opt. ALL 
            OptimizerPrefix = ("{}_OptGaussAll_Final".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
                        
        #*******************************************************************************************#             
        # End Option2
        
        
        # Option3
        #*******************************************************************************************#
        if GaussMethod == 3: 
           
            for gID in range(NumberGaussians):
                for SysFF in SysFFList: #Loop through all system FFs
                    FFList = SysFF[0] # Contains Bond and/or Splines
                    FFGaussians = SysFF[1] # Constains Gaussians
                    
                    if UseLocalDensity:
                        PLD = FFList[1]
                    
                    for index, PGauss in enumerate(FFGaussians):
                        PGauss.FreezeSpecificParam([0,1,4]) # Always Freeze LJ and Gauss offsets
			PGauss.MinHistFrac = 0.01
                        if gID == 0: # on the first gaussian; opt this one alone for now
                            OptimizerPrefix = ("{}_OptGauss{}".format(SrelName,gID))
                            
                        
                        else:
                            OptimizerPrefix = ("{}_OptGauss{}and{}".format(SrelName,(gID-1),gID))
                            if gID == index or gID == index-1: # unfix current and prior
                                PGauss.B.Fixed = False
                                PGauss.Kappa.Fixed = False
                            else:
                                PGauss.B.Fixed = True
                                PGauss.Kappa.Fixed = True
                        
                # opt. 
                RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
            
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
            
                for index, PGauss in enumerate(FFGaussians): # Unfix all parameters for final opt.
                    PGauss.B.Fixed = False
                    PGauss.Kappa.Fixed = False
            
            # opt. ALL 
            OptimizerPrefix = ("{}_OptGaussAll_Final".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)            

        #*******************************************************************************************#       
        # End Option3


        # Option4
        #*******************************************************************************************#
        if GaussMethod == 4:
            
            # opt. 
            OptimizerPrefix = ("{}_LSQFit_OptGaussAll_Final".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
            
        #*******************************************************************************************#    
        # End Option4
        
        
        # Option5
        #*******************************************************************************************#
        if GaussMethod == 5: 
             
            for gID in range(NumberGaussians):
                for SysFF in SysFFList: #Loop through all system FFs
                    FFList = SysFF[0] # Contains Bond and/or Splines
                    FFGaussians = SysFF[1] # Constains Gaussians
                    
                    PBond = FFList[0]
                    if UseLocalDensity:
                        PLD = FFList[1]
                    
                    for index, PGauss in enumerate(FFGaussians):
                        if gID == index: # unfix parameters
                            PGauss.B.Fixed = False
                            PGauss.Kappa.Fixed = False
                        else:
                            PGauss.B.Fixed = True
                            PGauss.Kappa.Fixed = True
                        
                # opt. 
                OptimizerPrefix = ("{}_LSQFit_OptGauss{}".format(SrelName,gID))
                RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
            
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                    
                for index, PGauss in enumerate(FFGaussians):
                    PGauss.B.Fixed = False
                    PGauss.Kappa.Fixed = False
            
            # opt. ALL 
            OptimizerPrefix = ("{}_LSQFit_OptGaussAll_Final".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
                        
        #*******************************************************************************************#             
        # End Option5
        
        
        # Option6
        #*******************************************************************************************#
        if GaussMethod == 6: 
             
            # opt. all gausian B's  
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                
                PBond = FFList[0]
                if UseLocalDensity:
                    PLD = FFList[1]
           
                for index, PGauss in enumerate(FFGaussians):
                        PGauss.B.Fixed = False
                        PGauss.Kappa.Fixed = True
                        
            # opt. all gausian B's  
            OptimizerPrefix = ("{}_LSQFit_OptALLBs".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
            
            # opt. all gausian Kappa's  
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                
                PBond = FFList[0]
                if UseLocalDensity:
                    PLD = FFList[1]
           
                for index, PGauss in enumerate(FFGaussians):
                        PGauss.B.Fixed = True
                        PGauss.Kappa.Fixed = False
                        
            # opt. all gausian Kappa's  
            OptimizerPrefix = ("{}_LSQFit_OptALLKappas".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
            
            # opt. all parameters together
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                    
                for index, PGauss in enumerate(FFGaussians):
                    PGauss.B.Fixed = False
                    PGauss.Kappa.Fixed = False
            
            # opt. ALL 
            OptimizerPrefix = ("{}_LSQFit_OptGaussAll_Final".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
                        
        #*******************************************************************************************#             
        # End Option6
        
        
        # Option7
        #*******************************************************************************************#
        if GaussMethod == 7: 
            FracMin = 0.75
            FracMax = 1.25
            
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                
                PBond = FFList[0]
                PBond.FConst.Fixed = False
                PBond.FConst.min = FracMin*PBond.FConst
                PBond.FConst.max = FracMax*PBond.FConst
                
                if UseLocalDensity:
                    PLD = FFList[1]
           
                for index, PGauss in enumerate(FFGaussians):
                        PGauss.B.Fixed = False
                        PGauss.Kappa.Fixed = False
                        PGauss.B.min = FracMin*PGauss.B
                        PGauss.B.max = FracMax*PGauss.B
                        PGauss.Kappa.min = FracMin*PGauss.Kappa
                        PGauss.Kappa.max = FracMax*PGauss.Kappa
                        
            # opt. all
            OptimizerPrefix = ("{}_LSQFit_OptALL_Final".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
                        
        #*******************************************************************************************#             
        # End Option7
        
        
        # Option8
        #*******************************************************************************************#
        if GaussMethod == 8: 
            FracMin = 0.75
            FracMax = 1.25
            
            # opt. all gausian B's  
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                
                PBond = FFList[0]
                if UseLocalDensity:
                    PLD = FFList[1]
           
                for index, PGauss in enumerate(FFGaussians):
                        PGauss.B.Fixed = False
                        PGauss.Kappa.Fixed = True
                        
            # opt. all gausian B's  
            OptimizerPrefix = ("{}_LSQFit_OptALLBs".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
            
            # opt. all gausian Kappa's  
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                
                PBond = FFList[0]
                if UseLocalDensity:
                    PLD = FFList[1]
           
                for index, PGauss in enumerate(FFGaussians):
                        PGauss.B.Fixed = True
                        PGauss.Kappa.Fixed = False
                        
            # opt. all gausian Kappa's  
            OptimizerPrefix = ("{}_LSQFit_OptALLKappas".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
                                    
            # opt. all parameters, but with constraints on all parameters
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                
                PBond = FFList[0]
                PBond.FConst.Fixed = False
                PBond.FConst.min = FracMin*PBond.FConst
                PBond.FConst.max = FracMax*PBond.FConst
                
                if UseLocalDensity:
                    PLD = FFList[1]
           
                for index, PGauss in enumerate(FFGaussians):
                        PGauss.B.Fixed = False
                        PGauss.Kappa.Fixed = False
                        PGauss.B.min = FracMin*PGauss.B
                        PGauss.B.max = FracMax*PGauss.B
                        PGauss.Kappa.min = FracMin*PGauss.Kappa
                        PGauss.Kappa.max = FracMax*PGauss.Kappa
                        
            # opt. all parameters
            OptimizerPrefix = ("{}_LSQFit_OptALL_Final".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
                        
        #*******************************************************************************************#             
        # End Option8
 
        # Option9
        #*******************************************************************************************#
        if GaussMethod == 9: 
            FracMin = 0.999
            FracMax = 1.001
            
            # opt. all gausian B's  
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                
                PBond = FFList[0]
                if UseLocalDensity:
                    PLD = FFList[1]
           
                for index, PGauss in enumerate(FFGaussians):
                        PGauss.B.Fixed = False
                        PGauss.Kappa.Fixed = True
                        
            # opt. all gausian B's  
            OptimizerPrefix = ("{}_LSQFit_OptALLBs".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
                                    
            # opt. all parameters, but with constraints on all parameters
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                
                PBond = FFList[0]
                PBond.FConst.Fixed = False
                PBond.FConst.min = FracMin*PBond.FConst
                PBond.FConst.max = FracMax*PBond.FConst
                
                if UseLocalDensity:
                    PLD = FFList[1]
           
                for index, PGauss in enumerate(FFGaussians):
                        PGauss.B.Fixed = False
                        PGauss.Kappa.Fixed = False
                        PGauss.B.min = FracMin*PGauss.B
                        PGauss.B.max = FracMax*PGauss.B
                        PGauss.Kappa.min = FracMin*PGauss.Kappa
                        PGauss.Kappa.max = FracMax*PGauss.Kappa
                        
            # opt. all parameters
            OptimizerPrefix = ("{}_LSQFit_OptALL_Final".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
                        
        #*******************************************************************************************#             
        # End Option9
        
        # Option10
        #*******************************************************************************************#
        if GaussMethod == 10: 
            
            # opt. all gausian B's  
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                
                PBond = FFList[0]
                if UseLocalDensity:
                    PLD = FFList[1]
           
                for index, PGauss in enumerate(FFGaussians):
                        PGauss.B.Fixed = False
                        PGauss.Kappa.Fixed = True
                        
            # opt. all gausian B's  
            OptimizerPrefix = ("{}_LSQFit_OptALLBs".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)
                                    
            # opt. all gausian Kappa's  
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                
                PBond = FFList[0]
                if UseLocalDensity:
                    PLD = FFList[1]
           
                for index, PGauss in enumerate(FFGaussians):
                        PGauss.B.Fixed = True
                        PGauss.Kappa.Fixed = False
                        
            # opt. all gausian Kappa's  
            OptimizerPrefix = ("{}_LSQFit_OptALLKappas_Final".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs, MaxIter=None, SteepestIter=0)                       
                        
        #*******************************************************************************************#             
        # End Option10
# output final gaussian potential
if GaussMethod in {4,5,6,7,8,9,10}:
    
    r_max = Cut
    r_min = 0.00001
    
    rs = opt[4]
    u_gauss = opt[2][NumberGaussians-1]
    u_spline = opt[3]
    distances = np.linspace(r_min,r_max,1000)
    
    u_pot_final = []
    SysFF = SysFFList[0] # Just pick first 
    print('\nForce-fields being used to calculate final Gaussian basis set.')
    print(SysFF[1])
    for rij in distances:
        val_temp = 0
        for FF in SysFF[1]:
            val_temp += FF.Val(rij)
        u_pot_final.append(val_temp)
    
    np.savetxt('u_pot_final.data',zip(distances,np.asarray(u_pot_final))) 
    
    Knots = [float(i) for i in re.split(' |,',SplineKnots) if len(i)>0]
    
    plt.figure()
    plt.plot(rs,u_spline,label="spline",linewidth = 3)
    plt.plot(rs,u_gauss,label="{}-Gaussian".format(NumberGaussians),linewidth = 3)
    plt.plot(distances,u_pot_final,label="Relaxed_{}-Gaussian".format(NumberGaussians),linewidth = 3)
    rs_knots = np.linspace(0,Cut,(NSplineKnots))
    plt.scatter(rs_knots,Knots,label = "spline knots",c='r')
    plt.ylim(min(np.min(u_spline),np.min(u_gauss), np.min(u_pot_final))*2,4)
    plt.xlim(0,Cut)
    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.legend(loc='best')
    plt.savefig('FinalGaussFit.pdf')  
    plt.close()
        
''' ***************************************************************** '''
''' Run the converged CG model to calculate Rg, Pressure, etc....     '''
''' ***************************************************************** '''                
             
if RunConvergedCGModel:
    UseLammpsMD     		= True
    UseSim          		= False
    CalcPress       		= True
    CalcRg          		= True
    CalcRee         		= True
    OutputDCD       		= True
    CalcStatistics 			= True
    ReturnTraj     			= False
    CheckTimestep  			= True
    PBC             		= True
    CnvDATA2PDB				= True	# Convert lammps.data to .pdb using MDAnalysis
    Make_molecules_whole    = True  # Uses MDTraj to make molecules whole
    SaveCalculationsToFile 	= True

    import MDAnalysis as mda
    import mdtraj as md
    from pymbar import timeseries
    from HistogramTools import HistogramRee
    import stats_TrajParse as stats_mod
    import stats

    print (SysList)
    for Sys_Index, Sys in enumerate(SysList):
        print('Running molecular dynamics on converged CG model: {}'.format(Sys.Name))
        os.mkdir(Sys.Name+'_PressData')
        
        # Read in force-field
        for subdir, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if 'Final_ff' in file:
                    ff_file = file
                    print('The force-field file is {}'.format(ff_file))
        if ff_file is None:
            print('No converged force-field file found {}'.format(ff_file))
        
        else:
            with open(ff_file, 'r') as of: s = of.read()
            Sys.ForceField.SetParamString(s)
            
            fileobject = open('measures_Sys.dat','w')
            
            Sys.TempSet = 1.0
            TempSet = 1.0

            #initial positions and velocities
            sim.system.positions.CubicLatticeFill(Sys, Random = 0.1)
            sim.system.velocities.Canonical(Sys, Temp = TempSet)
                        
            #configure integrator
            Int = Sys.Int
            
            # MD iterations
            NStepsMin = NSteps_Min_DUMMY
            NStepsEquil = NSteps_Equil_DUMMY
            NStepsProd = NSteps_Prod_DUMMY
            WriteFreq = WriteFreq_DUMMY
            
            
            if ScaleRuns:
                temp_StepsEquil = NStepsEquil*RunStepScaleList[Sys_Index]
                temp_StepsProd  = NStepsProd*RunStepScaleList[Sys_Index]
            else:
                temp_StepsEquil  = NStepsEquil
                temp_StepsProd   = NStepsProd

            Int.Method = Int.Methods.VVIntegrate
            Int.Method.Thermostat = Int.Method.ThermostatLangevin
    	    Int.Method.TimeStep = TimeStep_DUMMY # note: reduced units
            Int.Method.LangevinGamma = 1/(100*Int.Method.TimeStep)

            if UseSim:
                fobj = open('measures_{}.dat'.format(Sys_Index), 'w')
                Sys.Measures.VerboseOutput(fobj = fobj, StepFreq=5000)                                                                                                                                           
                print "Now conducting warm-up...."
                Int.Run(temp_StepsEquil)
                #Sys.Measures.Reset()
                print "Now running production runs...."
                Int.Run(temp_StepsProd)
                print "timing:", Int.TimeElapsed
                print "\n"
            
            if UseLammpsMD:
                if OutputDCD:
                    TrajFile = 'production_{}.dcd'.format(Sys_Index)
                else:
                    TrajFile = 'production_{}.lammpstrj'.format(Sys_Index)
                
                ret = sim.export.lammps.MakeLammpsTraj(Sys, DelTempFiles = False, Prefix = Sys.Name+'_', TrajFile = TrajFile, 
                                                       Verbose = True, NStepsMin = NStepsMin, NStepsEquil = temp_StepsEquil, NStepsProd = temp_StepsProd,
                                                       WriteFreq = WriteFreq, CalcPress = CalcPress, OutputDCD = OutputDCD, ReturnTraj = ReturnTraj, Nevery=500, Nrepeat=1, Nfreq=500)
          
            if CheckTimestep:
                # For final check of the timestep just for check
                sim.integrate.velverlet.FindTimeStep(Sys, NSteps = 10000, EneFracError = 1.0e-3, Verbose=True)
           
            ### **************************************************************************** ###
            ### ***************************** STATISTICS *********************************** ###
            ### **************************************************************************** ###
            
            if CalcStatistics:
                def statistics_py(DataFilename,Col):
                    ''' Do data using stats.py '''
                    # Calculate using stats.py
                    f = open(DataFilename,"rw")
                    warmupdata, proddata, idx = stats.autoWarmupMSER(f,Col)
                    nsamples,(min,max),mean,semcc,kappa,unbiasedvar,autocor = stats.doStats(warmupdata,proddata)

                    return ([nsamples,(min,max),mean,semcc,kappa,unbiasedvar,autocor])
                    
                def pymbar_statistics(dataset):
                    ''' Do PyMbar Analysis on the data using timeseries '''
                    dataset = np.asarray(dataset).flatten()
                    dataset_temp = dataset
                    pymbar_timeseries = (timeseries.detectEquilibration(dataset)) # outputs [t0, g, Neff_max] 
                    t0 = pymbar_timeseries[0] # the equilibrium starting indices
                    Data_equilibrium = dataset#[t0:]
                    g = pymbar_timeseries[1] # the statistical inefficiency, like correlation time
                    indices = timeseries.subsampleCorrelatedData(Data_equilibrium, g=g) # get the indices of decorrelated data
                    dataset = Data_equilibrium[indices] # Decorrelated Equilibrated data
                    dataset = np.asarray(dataset)
                    P_autocorrelation = timeseries.normalizedFluctuationCorrelationFunction(dataset, dataset, N_max=None, norm=True)
                    # Calculate using stats.py
                    warmupdata = 0
                    ''' Use stats.py to calculate averages '''
                    nsamples,(min,max),mean,semcc,kappa,unbiasedvar,autocor = stats_mod.doStats(warmupdata,dataset_temp)

                    return ([np.mean(dataset),np.var(dataset),np.sqrt(np.divide(np.var(dataset),len(dataset))),len(indices), len(Data_equilibrium), g, mean, semcc, kappa])
                
                import stats_TrajParse as stats
                # Walk through and find trajectory for specific system
                for subdir, dirs, files in os.walk(os.getcwd()):
                    for file in files:
                        if '.dcd' in file and str(Sys.Name) in file: # look for the current trajectory file
                            LammpsData  = Sys.Name+'_lammps.data'
                            LammpsTrj   = Sys.Name+'_'+TrajFile
                            print('*************')
                            print('**** TRJ ****')
                            print('*************')
                            print(LammpsTrj)
                            print (LammpsData)
                            
                            SaveFilename = Sys.Name+'_RgReeData'
                            
                            if SaveCalculationsToFile: os.mkdir(SaveFilename)
                            
                            if CnvDATA2PDB:
                                # Converts LAMMPS.data to .pdb with structure information
                                u = mda.Universe(LammpsData)
                                gr = u.atoms
                                gr.write(Sys.Name+'.pdb')
                                top_file = Sys.Name+'.pdb'
                            
                            ''' Load in trajectory file '''
                            traj = md.load(LammpsTrj,top=top_file)#, atom_indices=atoms_list)
                            
                            print ("Unit cell:")
                            print ("	{}".format(traj.unitcell_lengths[0])) # List of the unit cell on each frame
                            print ('Number of frames:')
                            print ("	{}".format(traj.n_frames))
                            print ('Number of molecule types:')
                            print ("	{}".format(traj.n_chains))
                            print ('Number of molecules:')
                            print ("	{}".format(traj.n_residues))
                            numberpolymers = int(traj.n_residues)
                            MoleculeResidueList = range(0,numberpolymers) # important for Rg calculation
                            print ('Number of atoms:')
                            print ("	{}".format(traj.n_atoms))
                            DOP = int(traj.n_atoms/numberpolymers)
                            print ("Atom 1 coordinates:")
                            print ('	{}'.format(traj.xyz[0][0]))
                            print ('Number of polymers:')
                            print ("	{}".format(numberpolymers))
                            print ('Degree of polymerization:')
                            print ("	{}".format(DOP))
                            
                            # Get atom ends for Ree
                            cnt = 0
                            ReeAtomIndices = [] # remember atom ID - 1 is the atom index
                            temp = []
                            for i in range(numberpolymers*DOP):
                                if cnt == 0:
                                    temp.append(i)
                                    cnt += 1
                                elif cnt == (DOP-1):
                                    temp.append(i)
                                    ReeAtomIndices.append(temp)
                                    temp = []
                                    cnt = 0
                                else:
                                    cnt += 1
                                    
                            print(ReeAtomIndices)
                            
                            if Make_molecules_whole: # generates bonding list for each molecule
                                bonds = []
                                cnt = 0
                                for i in range(numberpolymers*DOP):
                                    cnt += 1
                                    if cnt == (DOP):
                                        pass
                                        cnt = 0
                                    else:
                                        bonds_temp = [i,i+1]
                                        bonds.append(bonds_temp)
                                if CnvDATA2PDB: # Automatically finds the bonds from the topology file
                                    bonds = None
                                else:
                                    bonds = np.asarray(bonds,dtype=np.int32)
                                traj.make_molecules_whole(inplace=True, sorted_bonds=bonds)
                            
                            if SaveCalculationsToFile == True: os.chdir(SaveFilename)
                            
                            ReeTimeseries = []
                            RgTimeseries = []
                            Ree_averages = []
                            Rg_avg_stats = []
                            Rg_averages = []
                            Ree_avg_stats = []
                            ''' Calculate Radius-of-gyration '''
                            if CalcRg:
                                print('Calculating radius-of-gyration...')
                                # Compute the radius-of-gyration
                                ElementDictionary ={
                                            "carbon": 12.01,
                                            "hydrogen": 1.008,
                                            "oxygen": 16.00,
                                            "nitrogen": 14.001,
                                            "virtual site": 1.0,
                                            "virtual_site": 1.0,
                                            "sodium": "na+",
                                            }

                                Rg_list = []
                                Rg_Ave_list = []

                                for i,molecule in enumerate(MoleculeResidueList):
                                    atom_indices = traj.topology.select('resid {}'.format(i)) #and (resname UNL) or (resneme LEF) or (resname RIG)
                                    mass_list = []
                                    for index in atom_indices:
                                        temp = ElementDictionary[str(traj.topology.atom(index).element)]
                                        mass_list.append(temp)
                                    
                                    print ('Number of atoms in molecule {}'.format(i))
                                    print ('	{}'.format(len(atom_indices)))
                                    Rg = md.compute_rg(traj.atom_slice(atom_indices),np.asarray(mass_list))
                                    RgTimeseries.append(Rg)
                                    
                                    np.savetxt('Rg_out_mdtraj_molecule_{}.dat'.format((i)), Rg)

                                    stats_out = pymbar_statistics(Rg)
                                    
                                    RgAvg = stats_out[0]
                                    RgVariance = stats_out[2]**2
                                    CorrTime = stats_out[5]
                                    Rg_averages.append([RgAvg,RgVariance,CorrTime])
                                    Rg_avg_stats.append([stats_out[6],stats_out[7],stats_out[8]])
                                    
                                    print ('The radius of gyration for molecule {} is:'.format(i))
                                    print ('	{0:2.4f} +/- {1:2.5f}'.format(RgAvg,np.sqrt(RgVariance)))
                                    
                                    ''' Plot the radius of gyration '''
                                    plt.plot(Rg, "k-")
                                    plt.xlabel('timestep')
                                    plt.ylabel('Radius-of-gryation')
                                    plt.savefig("Rg_molecule_{}.pdf".format(i),bbox_inches='tight')
                                    plt.close()
                        
                            if CalcRee:
                                print('Calculating end-to-end distance...')
                                for i,temp_pair in enumerate(ReeAtomIndices):
                                    EndEndDist = md.compute_distances(traj,atom_pairs=[temp_pair], periodic=False, opt=True)
                                    ReeTimeseries.append(EndEndDist)
                                    
                                    stats_out = pymbar_statistics(EndEndDist)
                                    
                                    ReeAvg = stats_out[0]
                                    ReeVariance = stats_out[2]**2
                                    CorrTime = stats_out[5]
                                    Ree_averages.append([ReeAvg,ReeVariance,CorrTime])
                                    Ree_avg_stats.append([stats_out[6],stats_out[7],stats_out[8]])
                                    
                                    print ('The End-end distance for molecule {} is:'.format(i))
                                    print ('	{0:2.4f} +/- {1:2.5f}'.format(ReeAvg,np.sqrt(ReeVariance)))
                                    
                                    ''' Plot the Ree '''
                                    plt.plot(EndEndDist, "k-")
                                    plt.xlabel('timestep')
                                    plt.ylabel('Ree')
                                    plt.savefig("Ree_{}.pdf".format(i),bbox_inches='tight')
                                    plt.close()

                                    np.savetxt('Ree_{}.dat'.format(i), EndEndDist)
                            
                            # Move backup to working directory
                            if SaveCalculationsToFile == True: os.chdir("..")
                        
                            # Continue saving histograms to directories
                            if SaveCalculationsToFile == True: os.chdir(SaveFilename)
                            
                            if CalcRee:
                                ''' Histogram Ree '''
                                Ree_temp = []
                                for i in ReeTimeseries:
                                    Ree_temp.extend(i)
                                Ree_data = np.asarray(Ree_temp)
#                                HistogramRee(Ree_data, number_bins=25, DoBootStrapping=True, ShowFigures=False, NormHistByMax=True, 
#                                                    TrimRee=False, ReeCutoff=1.5, ReeMinimumHistBin=0., scale=1., gaussian_filter=False, sigma=2 )

                                ''' Plot all the Ree '''
                                for EndEndDist in ReeTimeseries:
                                    plt.plot(EndEndDist)
                                    plt.xlabel('timestep A.U.')
                                    plt.ylabel('Ree')
                                plt.savefig("Ree_total.pdf",bbox_inches='tight')
                                plt.close()
                        
                            if CalcRg:
                                ''' Plot all the Rg '''
                                for RgDist in RgTimeseries:
                                    plt.plot(RgDist)
                                    plt.xlabel('timestep A.U.')
                                    plt.ylabel('Rg')
                                plt.savefig("Rg_total.pdf",bbox_inches='tight')
                                plt.close()

                            print ('****** TOTALS *******')

                            ''' Calculate the ensemble averages '''
                            stats_out = open('stats_out.data','w')
                            if CalcRee:
                                ReeTotal = 0
                                Stats_ReeTotal = 0 
                                ReeVarianceTotal = 0
                                Stats_ReeVarTotal = 0 
                                CorrTime = []
                                Stats_CorrTime = []
                                for index, Ree in enumerate(Ree_averages):
                                    ReeTotal = ReeTotal + Ree[0]
                                    ReeVarianceTotal = ReeVarianceTotal + Ree[1]
                                    CorrTime.append(Ree[2])
                                    
                                    #from stats.py script
                                    Stats_ReeTotal = Stats_ReeTotal + Ree_avg_stats[index][0]
                                    Stats_ReeVarTotal = Stats_ReeVarTotal + (Ree_avg_stats[index][1])**2
                                    Stats_CorrTime.append(Ree_avg_stats[index][2])
                                    

                                ReeAverage = ReeTotal/len(Ree_averages)
                                ReeStdErr  = np.sqrt(ReeVarianceTotal/len(Ree_averages))
                                ReeAvgCorrTime = np.average(CorrTime)
                                ReeCorrTimeStdErr = np.sqrt(np.var(CorrTime)/len(CorrTime))
                                Stats_ReeAverage = Stats_ReeTotal/len(Ree_averages)
                                Stats_StdErr = np.sqrt(Stats_ReeVarTotal/len(Ree_averages))
                                Stats_AvgCorrTime = np.average(Stats_CorrTime)
                                Stats_CorrTimeStdErr = np.sqrt(np.var(Stats_CorrTime)/len(CorrTime))
                                print ('Total End-end distance average is: {0:4.4f} +/- {1:3.6f}'.format(ReeAverage,ReeStdErr))
                                print ('Total End-end distance avg. correlation time: {0:5.4f} +/- {1:5.6f}'.format(ReeAvgCorrTime, ReeCorrTimeStdErr))
                                print ('STATS: Total Ree distance avg is : {0:4.4f} +/- {1:3.6f}'.format(Stats_ReeAverage,Stats_StdErr))
                                print ('STATS: Total Ree Corr. Time avg is : {0:4.4f} +/- {1:3.6f}'.format(Stats_AvgCorrTime,Stats_CorrTimeStdErr))
                                stats_out.write('Total End-end distance average is: {0:4.4f} +/- {1:3.6f}\n'.format(ReeAverage,ReeStdErr))
                                stats_out.write('Total End-end distance avg. correlation time: {0:5.4f} +/- {1:5.6f}\n'.format(ReeAvgCorrTime, ReeCorrTimeStdErr))
                                stats_out.write('STATS: Total Ree distance avg is : {0:4.4f} +/- {1:3.6f}\n'.format(Stats_ReeAverage,Stats_StdErr))
                                stats_out.write('STATS: Total Ree Corr. Time avg is : {0:4.4f} +/- {1:3.6f}\n'.format(Stats_AvgCorrTime,Stats_CorrTimeStdErr))
                                
                            if CalcRg:
                                RgTotal = 0
                                Stats_RgTotal = 0
                                RgVarianceTotal = 0
                                Stats_RgVarTotal = 0
                                CorrTime = []
                                Stats_RgCorrTime = []
                                for index, Rg in enumerate(Rg_averages):
                                    RgTotal = RgTotal + Rg[0]
                                    RgVarianceTotal = RgVarianceTotal + Rg[1]
                                    CorrTime.append(Rg[2]) 
                                    
                                    #from stats.py script
                                    Stats_RgTotal = Stats_RgTotal + Rg_avg_stats[index][0]
                                    Stats_RgVarTotal = Stats_RgVarTotal + (Rg_avg_stats[index][1])**2
                                    Stats_RgCorrTime.append(Rg_avg_stats[index][2])
                                    
                                RgAverage = RgTotal/len(Rg_averages)
                                RgStdErr  = np.sqrt(RgVarianceTotal/len(Rg_averages))
                                RgAvgCorrTime = np.average(CorrTime)
                                RgCorrTimeStdErr = np.sqrt(np.var(CorrTime)/len(CorrTime))
                                Stats_RgAverage = Stats_RgTotal/len(Rg_averages)
                                Stats_RgStdErr = np.sqrt(Stats_RgVarTotal/len(Rg_averages))
                                Stats_AvgRgCorrTime = np.average(Stats_RgCorrTime)
                                Stats_RgCorrTimeStdErr = np.sqrt(np.var(Stats_RgCorrTime)/len(CorrTime))
                                print ('Total Rg average is: {0:2.3f} +/- {1:2.5f}'.format(RgAverage,RgStdErr))
                                print ('Total Rg avg. correlation time: {0:5.4f} +/- {1:5.6f}'.format(RgAvgCorrTime, RgCorrTimeStdErr))
                                print ('STATS: Total Rg distance avg is : {0:4.4f} +/- {1:3.6f}'.format(Stats_RgAverage,Stats_RgStdErr))
                                print ('STATS: Total Rg Corr. Time avg is : {0:4.4f} +/- {1:3.6f}'.format(Stats_AvgRgCorrTime,Stats_RgCorrTimeStdErr))
                                stats_out.write('Total Rg average is: {0:2.3f} +/- {1:2.5f}\n'.format(RgAverage,RgStdErr))
                                stats_out.write('Total Rg avg. correlation time: {0:5.4f} +/- {1:5.6f}\n'.format(RgAvgCorrTime, RgCorrTimeStdErr))
                                stats_out.write('STATS: Total Rg distance avg is : {0:4.4f} +/- {1:3.6f}\n'.format(Stats_RgAverage,Stats_RgStdErr))
                                stats_out.write('STATS: Total Rg Corr. Time avg is : {0:4.4f} +/- {1:3.6f}\n'.format(Stats_AvgRgCorrTime,Stats_RgCorrTimeStdErr))
                                
                                
                            # Calculate alpha value
                            if RgAverage != 0 and ReeAverage != 0:
                                alpha = ReeAverage/RgAverage
                                alpha_std = np.sqrt((1/RgAverage)**2*RgStdErr**2 + (ReeAverage/RgAverage**2)**2*ReeStdErr**2)
                                Stats_alpha = Stats_ReeAverage/Stats_RgAverage
                                Stats_alpha_std = np.sqrt((1/Stats_RgAverage)**2*Stats_RgStdErr**2 + (Stats_ReeAverage/Stats_RgAverage**2)**2*Stats_StdErr**2)
                                print ('The alpha value: Ree/Rg is: {0:4.4f} +/- {1:4.4f}'.format(alpha,alpha_std))
                                print ('STATS: The alpha value: Ree/Rg is: {0:4.4f} +/- {1:4.4f}'.format(Stats_alpha,Stats_alpha_std))
                                stats_out.write('The alpha value: Ree/Rg is: {0:4.4f} +/- {1:4.4f}\n'.format(alpha,alpha_std))
                                stats_out.write('STATS: The alpha value: Ree/Rg is: {0:4.4f} +/- {1:4.4f}\n'.format(Stats_alpha,Stats_alpha_std))
                                
                            # Move backup to working directory
                            if SaveCalculationsToFile == True: os.chdir("..")
                

                ''' Calculate Pressure '''
                if CalcPress:
                    print("Calculating Pressure Statistics...")
                    PressMean          = []
                    PressVar           = []
                    PressSamples       = []
                    PressCorr          = []
                    samplecount        = 0
                    
                    for subdir, dirs, files in os.walk(os.path.join(os.getcwd(),Sys.Name+'_PressData')):
                        for file in files:
                            stats_out = statistics_py(os.path.join(os.getcwd(),Sys.Name+'_PressData',file),1)
                            PressMean.append(stats_out[2])
                            PressVar.append(stats_out[5])
                            PressCorr.append(stats_out[4])
                            PressStdErr = stats_out[3]
                    
                    PressCorrAvg = np.average(PressCorr)
                    PressCorrVar = np.var(PressCorr)
                    PressCorrStdErr = np.sqrt(PressCorrVar/len(PressCorr))
                    PressAvg = np.average(PressMean)
                    PressVar = np.sum(PressVar)
                    PressStdErr = PressStdErr

                
                
                with open(Sys.Name+'_Statistics.dat','w') as g:
                    g.write('#  Avg.    Pseudo-Var.    StdErr.     Corr.   Var.    StdErr.\n')
                    if CalcRg:
                        g.write('Rg        {0:8.4f}      {1:8.6f}      {2:8.6f}      {3:8.4f}      {4:8.6f}      {5:8.6f}\n'.format(Stats_RgAverage,Stats_RgVarTotal,Stats_RgStdErr,Stats_AvgRgCorrTime,np.var(Stats_RgCorrTime),Stats_RgCorrTimeStdErr))
                    if CalcRee:
                        g.write('Ree       {0:8.4f}      {1:8.6f}      {2:8.6f}      {3:8.4f}      {4:8.6f}      {5:8.6f}\n'.format(Stats_ReeAverage,Stats_ReeVarTotal,Stats_StdErr,Stats_AvgCorrTime,np.var(Stats_CorrTime),Stats_CorrTimeStdErr))
                    if CalcPress:
                        g.write('Press     {0:8.4e}      {1:8.4e}      {2:8.4e}      {3:8.4e}      {4:8.4e}      {5:8.4e}\n'.format(PressAvg,PressVar,PressStdErr,PressCorrAvg,PressCorrVar,PressCorrStdErr))

