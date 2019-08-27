#usr/bin/env python

### Testing for potentials in SIM suite.
### coded by TS

import os, numpy as np, time, cPickle as pickle
import sim, pickleTraj

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

# default settings
sim.export.lammps.InnerCutoff = 0.00001
sim.export.lammps.LammpsExec = 'lmp_omp'
sim.export.lammps.OMP_NumThread = Threads_DUMMY
sim.srel.optimizetrajlammps.LammpsDelTempFiles = False

sim.export.omm.platformName = 'OpenCL' # or 'OpenCL' or 'GPU' or 'CUDA'
sim.export.omm.device = -1 #-1 is default, let openmm choose its own platform.
sim.export.omm.NPairPotentialKnots = 500 #number of points used to spline-interpolate the potential
sim.export.omm.InnerCutoff = 0.001 #0.001 is default. Note that a small value is not necessary, like in the lammps export, because the omm export interpolates to zero
sim.srel.optimizetrajomm.OpenMMStepsMin = 0 #number of steps to minimize structure, 0 is default
sim.srel.optimizetrajomm.OpenMMDelTempFiles = False #False is Default
sim.export.omm.UseTabulated = True

#External potential
Ext = {"UConst": UConst_DUMMY, "NPeriods": NPeriods_DUMMY, "PlaneAxis": PlaneAxis_DUMMY, "PlaneLoc": PlaneLoc_DUMMY}
if Ext["UConst"] > 0:
    print("Using external sinusoid with UConst {}".format(Ext["UConst"]))
    UseExternal = True
else:
    UseExternal = False

# md iterations
StepsEquil = StepsEquil_DUMMY
StepsProd = StepsProd_DUMMY
StepsStride = StepsStride_DUMMY
RunStepScaleList = RunStepScaleList_DUMMY
GaussMethod = GaussMethod_DUMMY

Cut = Cut_DUMMY
FixBondDist0 = True
PBondDist0 = 0. # For zero centered bonds set to 0.
UseLocalDensity = False
CoordMin = 0    
CoordMax = 350
LDKnots    = 10
# N.S. TODO:
# Change how Gaussian potentials are handled

NumberGaussians = NumberGaussians_DUMMY
ScaleRuns = ScaleRuns_DUMMY

RunSpline = RunSpline_DUMMY
SplineKnots = SplineKnots_DUMMY

# Spline options
#   Option1 = Constant slope 
#   Option2 = Constant slope, then turn-off
#   Option3 = Slope unconstrained

SplineConstSlope = SplineConstSlope_DUMMY # Turns on Constant slope for first opt.; then shuts it off for final opt. 
FitSpline = FitSpline_DUMMY # Turns on Gaussian Fit of the spline for the initial guess

# N.S. TODO:
# Add in option to specify the Spline inner slope (i.e. 2kbTperA)
# Add in Spline fit parameters (i.e. make stronger or longer ranged based on mapping)
SysLoadFF = SysLoadFF_DUMMY # Use if you desire to seed a run with an already converged force-field.
force_field_file = force_field_file_DUMMY               
UseWPenalty = UseWPenalty_DUMMY
UseLammps = UseLammps_DUMMY 
UseOMM = UseOMM_DUMMY
UseSim = False
WriteTraj = True
UseExpandedEnsemble = UseExpandedEnsemble_DUMMY
RunConvergedCGModel = True # Run the converged ff file at the end (calculates P and Rg statistics), 
# outputs LAMMPS .dcd trajectory (this requires modification of the lammps.py export in sim. 


def FreezeParameters(System_List, Pot, Parameters):
    # - Pot is the index of the potential with parameters to freeze.
    # - Parmaters is a list of parameters to freeze in Pot.
    for index, Sys in enumerate(System_List):
        for P_index, Pot in enumerate(Sys.ForceField):
            if P_index == Pot:
                Pot.FreezeSpecificParam(Parameters) 
                

def CreateForceField(Sys, Cut, UseLocalDensity, CoordMin, CoordMax, LDKnots, RunSpline, SplineKnots, NumberGaussians):
    ''' Function that creates the system force-field. '''
    
    FFList = []
    FFGaussians = []
    AtomType = Sys.World[0][0] #since only have one atom type in the system
    
    ''' Add in potentials '''
    # Add PBond, Always assumed to be the first potential object!
    PBond = sim.potential.Bond(Sys, Filter = sim.atomselect.BondPairs,
                               Dist0 = 0., FConst = 500., Label = 'Bond')
    
    PBond.Param.Dist0.Min = 0.
    FFList.extend([PBond])
    
    ''' Add Splines '''
    if RunSpline:
        PSpline = sim.potential.PairSpline(Sys, Filter = sim.atomselect.Pairs, Cut = Cut,
                                           NKnot = SplineKnots, Label = 'Spline', 
                                           NonbondEneSlope = "40kTperA", BondEneSlope = "40kTperA")
        if FitSpline:
            Max = 50.
            decay = 0.5
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
        for g in range(NumberGaussians):
            temp_Label = 'LJGauss{}'.format(g)
            temp_Gauss = sim.potential.LJGaussian(Sys, Filter = sim.atomselect.Pairs, Cut = Cut,
                                 Sigma = 1.0, Epsilon = 0.0, B = 0., Kappa = 0.15,
                                 Dist0 = 0.0, Label = temp_Label)
            
            if g == 0: # initialize the first Guassian to be repulsive
                temp_Gauss.Param.B = 5.
                temp_Gauss.Param.B.min = 0.
                
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

    return FFList, FFGaussians

def CreateSystem(Name, BoxL, NumberMolecules, NumberMonomers, Cut, UseLocalDensity, CoordMin, CoordMax,
                    LDKnots, RunSpline, SplineKnots, NumberGaussians):
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
    
    FFList, FFGaussians = CreateForceField(Sys, Cut, UseLocalDensity, CoordMin, CoordMax, LDKnots, 
                                RunSpline, SplineKnots,
                                NumberGaussians) 
                                
    Sys.ForceField.extend(FFList)
    Sys.ForceField.extend(FFGaussians)
    

    #set up the histograms
    for P in Sys.ForceField:
        P.Arg.SetupHist(NBin = 8000, ReportNBin = 200)

    # lock and load
    Sys.Load()

    Sys.BoxL[:] = BoxL

    #initial positions and velocities
    sim.system.positions.CubicLattice(Sys)
    sim.system.velocities.Canonical(Sys, Temp = 1.0)
    Sys.TempSet = 1.0

    #configure integrator
    Int = Sys.Int

    Int.Method = Int.Methods.VVIntegrate        
    Int.Method.Thermostat = Int.Method.ThermostatLangevin
    Int.Method.TimeStep = 0.0001 # note: reduced units
    Int.Method.LangevinGamma = 1/(100*Int.Method.TimeStep)
    Sys.TempSet = TempSet
    
    return Sys, [FFList, FFGaussians]

    
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
    BoxL = Traj_Temp.FrameData['BoxL'][0] 
    if MappingRatio != 1:
        Traj_Temp = sim.traj.Mapped(Traj_Temp, Map, BoxL = BoxL)
        sim.traj.base.Convert(Traj_Temp, sim.traj.LammpsWrite, FileName = OutTraj, Verbose = True)
      
    
    SysTemp, [FFList, FFGaussians] = CreateSystem(Name, BoxL, NMol, CGDOP, Cut, UseLocalDensity, 
                                        CoordMin, CoordMax, LDKnots, RunSpline, SplineKnots, NumberGaussians)
    
    SysFFList.append([FFList, FFGaussians])
    
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
    
    
    # Freeze parameters that never change
    PBond = SysTemp.ForceField[0]
    if FixBondDist0:
        PBond.Dist0.Fixed = True
        PBond.Dist0 = PBondDist0
    
    if RunSpline == False:
        for g in range(NumberGaussians):
            PLJGauss_Temp = SysTemp.ForceField[g]
            PLJGauss_Temp.FreezeSpecificParam([0,1,4]) # Fix all LJ part to 0. and Gauss offset to 0.
            PLJGauss_Temp.MinHistFrac = 0.01         
    
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
	W = SysTemp.NDOF - 3*Press*BoxL**3
        Opt_temp.AddPenalty("Virial", W, MeasureScale = 1./SysTemp.NAtom, Coef = 1.e-80) #HERE also need to scale the measure by 1/NAtom to be comparable to Srel
        
    OptList.append(Opt_temp)
    SysList.append(SysTemp)
    NumAtomsList.append(SysTemp.NAtom)
    
''' ******************************************* '''
''' ********** Run Srel Optimization ********** '''
''' ******************************************* '''
RunOptimization = True

def RunSrelOptimization(Optimizer, OptimizerPrefix, UseWPenalty, StageCoefs):
    ''' Runs Srel Optimization '''
    Optimizer.FilePrefix = (OptimizerPrefix)
    if UseWPenalty == False:
        Optimizer.RunConjugateGradient()
    
    elif UseWPenalty == True:
        Optimizer.RunStages(StageCoefs = StageCoefs)


if RunOptimization:
    # Just always using the OptimizeMultiTrajClass
    Weights = [1.]*len(OptList)
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
        RunSrelOptimization(Optimizer, OptimizerPrefix, UseWPenalty, StageCoefs)
        
        
        if SplineOption == 'Option2':
            for SysFF in SysFFList:
                PSpline = SysFF[0][1]
		print "Relaxing spline knot constraints" 
                #PSpline.EneSlopeInner = None # Turn-off the EneSlopeInner
                PSpline.RelaxKnotConstraints() #relax all spline constraints 
            OptimizerPrefix = ("{}_OptSpline_Final".format(SrelName))
            
            # opt. 
            RunSrelOptimization(Optimizer, OptimizerPrefix, UseWPenalty, StageCoefs)

    else: # Run Gaussians optimization in stages
    # N.S. TODO: at somepoint could be useful to make this more robust using list so that an arbitrary number
    #                  of Gaussians can be added without manually adding in all the different stages/parameter fixing.

        ''' Run Guassians: There are 3 options currently.
        
            Option1: Run all together
            Option2: Stage one at a time; Run all together after individual
            Option3: Run First one, then allow prior and new one to float together; Run all together after individual
        
        '''
        
        # Option1
        #*******************************************************************************************#
        if GaussMethod == 1:
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                
                PBond = FFList[0]
                if FixBondDist0:
                    PBond.Dist0.Fixed = True
                    PBond.Dist0 = PBondDist0
                else: 
                    PBond.Dist0.Fixed = False
                    PBond.Dist0 = PBondDist0
                if UseLocalDensity:
                    PLD = FFList[1]
                
                for g, PGauss in enumerate(FFGaussians):
                    PGauss.FreezeSpecificParam([0,1,4]) # Freeze LJ and Gauss offsets
            
            # opt. 
            OptimizerPrefix = ("{}_OptGaussAll_Final".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs)
            
        #*******************************************************************************************#    
        # End Option1
        
        # Option2
        #*******************************************************************************************#
        if GaussMethod == 2: 
             
            for gID in range(NumberGaussians):
                for SysFF in SysFFList: #Loop through all system FFs
                    FFList = SysFF[0] # Contains Bond and/or Splines
                    FFGaussians = SysFF[1] # Constains Gaussians
                    
                    PBond = FFList[0]
                    if FixBondDist0:
                        PBond.Dist0.Fixed = True
                        PBond.Dist0 = PBondDist0
                    else: 
                        PBond.Dist0.Fixed = False
                        PBond.Dist0 = PBondDist0
                    if UseLocalDensity:
                        PLD = FFList[1]
                    
                    for index, PGauss in enumerate(FFGaussians):
                        PGauss.FreezeSpecificParam([0,1,4]) # Always Freeze LJ and Gauss offsets
                        if gID == index: # unfix parameters
                            PGauss.B.Fixed = False
                            PGauss.Kappa.Fixed = False
                        else:
                            PGauss.B.Fixed = True
                            PGauss.Kappa.Fixed = True
                        
                # opt. 
                OptimizerPrefix = ("{}_OptGauss{}".format(SrelName,gID))
                RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs)
            
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
                    
                for index, PGauss in enumerate(FFGaussians):
                    PGauss.B.Fixed = False
                    PGauss.Kappa.Fixed = False
            
            # opt. ALL 
            OptimizerPrefix = ("{}_OptGaussAll_Final".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs)
                        
        #*******************************************************************************************#             
        # End Option2
        
        
        # Option3
        #*******************************************************************************************#
        if GaussMethod == 3: 
           
            for gID in range(NumberGaussians):
                for SysFF in SysFFList: #Loop through all system FFs
                    FFList = SysFF[0] # Contains Bond and/or Splines
                    FFGaussians = SysFF[1] # Constains Gaussians
                    
                    PBond = FFList[0]
                    if FixBondDist0:
                        PBond.Dist0.Fixed = True
                        PBond.Dist0 = PBondDist0
                    else: 
                        PBond.Dist0.Fixed = False
                        PBond.Dist0 = PBondDist0
                    if UseLocalDensity:
                        PLD = FFList[1]
                    
                    for index, PGauss in enumerate(FFGaussians):
                        PGauss.FreezeSpecificParam([0,1,4]) # Always Freeze LJ and Gauss offsets
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
                RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs)
            
            for SysFF in SysFFList: #Loop through all system FFs
                FFList = SysFF[0] # Contains Bond and/or Splines
                FFGaussians = SysFF[1] # Constains Gaussians
            
                for index, PGauss in enumerate(FFGaussians): # Unfix all parameters for final opt.
                    PGauss.B.Fixed = False
                    PGauss.Kappa.Fixed = False
            
            # opt. ALL 
            OptimizerPrefix = ("{}_OptGaussAll_Final".format(SrelName))
            RunSrelOptimization(Optimizer,OptimizerPrefix, UseWPenalty, StageCoefs)            

        #*******************************************************************************************#       
        # End Option3


''' ***************************************************************** '''
''' Run the converged CG model to calculate Rg, Pressure, etc....     '''
''' ***************************************************************** '''                
             
if RunConvergedCGModel:
    UseLammpsMD     = True
    UseSim          = False
    CalcPress       = True
    CalcRg          = True
    CalcRee         = False
    OutputDCD       = True
    CalcStatistics  = True
    ReturnTraj      = False
    CheckTimestep   = True
    PBC             = True
    import MDAnalysis as mda
    import mdtraj as md
    
    print (SysList)
    for Sys_Index, Sys in enumerate(SysList):
        print('Running molecular dynamics on converged CG model: {}'.format(Sys.Name))
        os.mkdir(Sys.Name+'_PressData')
        os.mkdir(Sys.Name+'_RgData')
        
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
                temp_StepsEquil = NStepsEquil*RunStepScaleList[index]
                temp_StepsProd  = NStepsProd*RunStepScaleList[index]
            else:
                temp_StepsEquil  = NStepsEquil
                temp_StepsProd   = NStepsProd

            Int.Method = Int.Methods.VVIntegrate
            Int.Method.Thermostat = Int.Method.ThermostatLangevin
    	    Int.Method.TimeStep = 0.0001 # note: reduced units
	    Int.Method.LangevinGamma = 1/(100*Int.Method.TimeStep)

            if UseSim:
                print "Now conducting warm-up...."
                Int.Run(NStepsEquil)
                #Sys.Measures.Reset()
                print "Now running production runs...."
                Int.Run(NStepsProd)
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
                sim.integrate.velverlet.FindTimeStep(Sys, NSteps = 10000, EneFracError = 0.5e-5, Verbose=True)
           
           ### ***************************** STATISTICS *********************************** ###
            if CalcStatistics:
                def statistics_py(DataFilename,Col):
                    ''' Do data using stats.py '''
                    # Calculate using stats.py
                    f = open(DataFilename,"rw")
                    warmupdata, proddata, idx = stats.autoWarmupMSER(f,Col)
                    nsamples,(min,max),mean,semcc,kappa,unbiasedvar,autocor = stats.doStats(warmupdata,proddata)

                    return ([nsamples,(min,max),mean,semcc,kappa,unbiasedvar,autocor])
                
                import stats
                
                ''' Calculate Ree Data '''
                # More rhobust than Rg typically, since Rg will not be correct if PBC's not considered correctly
                
                if CalcRee:
                    print('Calculating end-to-end distance...')
                
                    for subdir, dirs, files in os.walk(os.getcwd()):
                        for file in files:
                            print(file)
                            if '.dcd' in file and str(Sys.Name) in file: # look for the current trajectory file
                                LammpsData  = Sys.Name+'_lammps.data'
                                LammpsTrj   = Sys.Name+'_'+TrajFile
               
                    EndEndDist = md.compute_distances(traj,atom_pairs=ReeAtomIndices, periodic=PBC, opt=True)
                    ReeTimeseries.append(EndEndDist)
                
                
                ''' Calculate Rg Data '''
                if CalcRg:
                    print("Calculating Rg Statistics...")
                    RgMean          = []
                    RgVar           = []
                    RgSamples       = []
                    RgCorr          = []
                    samplecount        = 0

                    for subdir, dirs, files in os.walk(os.getcwd()):
                        for file in files:
                            print(file)
                            if '.dcd' in file and str(Sys.Name) in file: # look for the current trajectory file
                                LammpsData  = Sys.Name+'_lammps.data'
                                LammpsTrj   = Sys.Name+'_'+TrajFile
                                print (LammpsData)
                                print('TRJ')
                                print(LammpsTrj)
                                u = mda.Universe(LammpsData,LammpsTrj)
                                
                                NMol = Sys.NMol
                                Atoms_Molecule = []
                                Header = []
                                Header.append('# Step')
                                for i in range(NMol):
                                    molID = i+1
                                    Atoms_Molecule.append(u.select_atoms("resid {}".format(molID)))
                                    Header.append('Rg_{}'.format(molID))
                                    
                                Rg_List = []
                                step = 0
                                for ts in u.trajectory:
                                    step += 1
                                    Rg_temp = []
                                    Rg_temp.append(step)
                                    for molecule in Atoms_Molecule:
                                        Rg_temp.append(molecule.radius_of_gyration())
                                    Rg_List.append(Rg_temp)
                                
                                Rg_Array = np.asarray(Rg_List)
                                np.savetxt(Sys.Name+'_Rg.data',Rg_Array,header='-'.join(Header))
                                
                                for i in range(NMol): # Calculate for each molecule
                                    molID = i+1 
                                    stats_out = statistics_py(os.path.join(os.getcwd(),(Sys.Name+'_Rg.data')),molID)
                                    RgMean.append(stats_out[2])
                                    RgVar.append(stats_out[5])
                                    RgCorr.append(stats_out[4])
                    
                    RgCorrAvg = np.average(RgCorr)
                    RgCorrVar = np.var(RgCorr)
                    RgCorrStdErr = np.sqrt(RgCorrVar/len(RgCorr))
                    RgAvg = np.average(RgMean)
                    RgVar = np.sum(RgVar)/NMol**2
                    RgStdErr = np.sqrt(RgVar*RgCorrAvg/step/NMol) # Neglected factoring in the standard error from RgCorrAvg.
                
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
                    PressCorrStdErr = np.sqrt(PressCorrVar/len(RgCorr))
                    PressAvg = np.average(PressMean)
                    PressVar = np.sum(PressVar)
                    PressStdErr = PressStdErr

                
                
                with open(Sys.Name+'_Statistics.dat','w') as g:
                    g.write('#  Avg.    Var.    StdErr.     Corr.   Var.    StdErr.\n')
                    if CalcRg:
                        g.write('Rg        {0:8.4f}      {1:8.6f}      {2:8.6f}      {3:8.4f}      {4:8.6f}      {5:8.6f}\n'.format(RgAvg,RgVar,RgStdErr,RgCorrAvg,RgCorrVar,RgCorrStdErr))
                    if CalcPress:
                        g.write('Press     {0:8.4e}      {1:8.4e}      {2:8.4e}      {3:8.4e}      {4:8.4e}      {5:8.4e}\n'.format(PressAvg,PressVar,PressStdErr,PressCorrAvg,PressCorrVar,PressCorrStdErr))

