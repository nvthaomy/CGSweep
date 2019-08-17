import time
import numpy as np
import scipy as sp
import scipy.stats
import math
import subprocess as prcs
import os, sys
from shutil import copyfile

''' USER-INPUT '''
TrajFileDir = 'traj'
CGModelScript = 'cgmodel_sweep_EE.py'
SubmitScriptName  = 'submit.sh'
SpecialName     = 'TESTRUNS'
NumberThreads = 1
JobRunTime = '700:00:00'
SplineKnots = 8
Cut = 10.
NMolList = [[1,4]] # Needs to be the list of # molecules if doing Exp. Ens. 
ScaleRuns = True 
RunStepScaleList = '[1.1,1]' # scales the CG runtime for systems in the NMolList, i.e. run dilute system longer
NumberGaussianBasisSets = [1,2,3]
CG_Mappings = [3,1]
RunSpline = True
RunGauss = False
#N.S. TODO: Add in option for EE runs
ExpEnsemble = True
GaussMethod = 1

''' USER-INPUT ''' 
''' Specify the trajectory list to use ''' 
# IF using ExpEnsemble = True need to make TrajList a list of list!

#TrajList = ['CG_AtomPos_np_02_T_320', 'CG_AtomPos_np_20_T_320']
TrajList = [['AAtraj_np01','AAtraj_np04']]
# parameter names and their values; need to specify trajectorylist above 
CGModel_ParameterNames = ['Cut','SplineKnots','ExpEnsemble','TrajList','Threads','NMol','RunStepScaleList','GaussMethod','ScaleRuns']
CGModel_Parameters     = [Cut,SplineKnots,ExpEnsemble,TrajList,NumberThreads,NMolList,RunStepScaleList,GaussMethod,ScaleRuns]


''' LESS USED DEFAULT OPTIONS'''
SaveCalculationsToFile = True
SaveFilename = '2019.07.24_CG_Run' # currently not functioning 


''' ************** Bulk of the Code ************************************* '''
''' ********************************************************************* '''

### FUNCTION DEFINITIONS USED BELOW ###

def BuildTrajList(ExpEnsemble, Traj):
    ''' Builds the Trajectory List and outputs a string representation of list. '''
    temp_TrajList = []
    if ExpEnsemble:
        for temp_trajname in Traj:
            temp_TrajList.append(str(temp_trajname+'.lammpstrj'))
    else:
        temp_TrajList.append(str(Traj+'.lammpstrj'))
    return temp_TrajList

def GenerateSubmitScript(CGModelScript, cwd, lines_SubmitScript, SubmitScriptName, NumberThreads, RunName, JobRunTime):
    ''' Generates the cluster submission script. '''
    
    lines_SubmitScript = lines_SubmitScript.replace('THREADS_DUMMY', str(NumberThreads))
    lines_SubmitScript = lines_SubmitScript.replace('JOBRUNTIME_DUMMY', str(JobRunTime))
    lines_SubmitScript = lines_SubmitScript.replace('NAME_DUMMY', str(RunName))
    lines_SubmitScript = lines_SubmitScript.replace('CGModelScript_DUMMY', str(CGModelScript))
    
    with open(str(SubmitScriptName), 'w') as g:
		g.write(lines_SubmitScript)
    
def CreateCGModelDirectory(ExpEnsemble, RunDirName,Traj,cwd,CGModel,CGModel_ParameterNames, CGModel_Parameters, 
                               CGMap, RunSpline, NumberGauss, SubmitScriptName, SubmitScript, NumberThreads, RunName, JobRunTime, TrajListInd = None):
    ''' Main function for creating the CG directory 
        
        FUNCTION-INPUTS:
            ExpEnsemble:            True or False, defined above.
            RunDirName:             The directory name for the folder to create for CG run.
            TrajName:               The trajectory list converted to string for replacement in CGModelScript.
            CGModel:                The CGModelScript loaded as a file object; 
                                        DUMMY variables replaced in their with parameters defined above.
            CGModel_ParameterNames: Prefixes of the DUMMY variables in the CGModelScript (i.e., NKnots_DUMMMY would be entered as NKnots)
            CGModel_Parameters:     The corresponding variables that replace these DUMMY variables in the script
            CGMap:                  The CG mapping for this run (i.e. 1,2,3,4) 
            RunSpline:              True or False, specifies whether to use spline for pair potential.
            NumberGauss:            The number of Gaussians to use in this run
            SubmitScriptName:       The submission script for the cluster 
            NumberThreads:          The number of threads to use for these jobs
            RunName:                The name of the job in the submission script
            JobRunTime:             The run time of the job in the submission script 
            TrajListInd:            Index of list of trajectories in TrajList for expanded ensemble. None for 1-state CG 
        FUNCTION-OUTPUTS:
            The function outputs a CG directory and submits this to the queue. 
    '''

   # make run dir.
    os.mkdir(RunDirName)
    # copy traj to run dir.
    if ExpEnsemble:
        source = os.path.join(cwd,TrajFileDir)
        #print (source)
        for subdir, dirs, files in os.walk(source):
            for file in files:
                #print (file)
                if file.endswith(".lammpstrj"):
                    #print(os.path.join(cwd,RunDirName,file))
                    copyfile(os.path.join(cwd,TrajFileDir,file),os.path.join(cwd,RunDirName,file))
                if "_ff" in file: # Incase one wants to seed run with FF file just put it in this directory
                    copyfile(os.path.join(cwd,TrajFileDir,file),os.path.join(cwd,RunDirName,file))
    else:
        source = os.path.join(cwd,TrajFileDir)
        #print(cwd)
        #print(os.path.join(cwd,TrajFileDir,str(Traj+'.lammpstrj')))
        #print(os.path.join(cwd,RunDirName)) 
        copyfile(os.path.join(cwd,TrajFileDir,str(Traj+'.lammpstrj')),os.path.join(cwd,RunDirName,str(Traj+'.lammpstrj')))
        for subdir, dirs, files in os.walk(source):
            for file in files:
                if "_ff" in file: # Incase one wants to seed run with FF file just put it in this directory
                    copyfile(os.path.join(cwd,TrajFileDir,file),os.path.join(cwd,RunDirName,file))
            
    # move into new directory
    os.chdir(RunDirName)
    
    # Generate the submission script in the directory
    GenerateSubmitScript(CGModelScript, cwd, SubmitScript, SubmitScriptName, NumberThreads, RunName, JobRunTime)
    
    # Replace the parameters in the cgmodel file
    temp_CGModel = CGModel
    for index, param_name in enumerate(CGModel_ParameterNames):
        param_value = CGModel_Parameters[index]
        if 'TrajList' in param_name:
            param_value = BuildTrajList(ExpEnsemble, Traj)
        if 'NMol' in param_name:
	    print ('NMol: {}'.format(param_value))
            if ExpEnsemble:
                param_value = str(param_value[TrajListInd])
            else:
                for NMol_val in param_value:
                    #print (Traj)
                    #print ('np_{:02d}'.format(NMol_val))
                    if ('np_{:02d}'.format(NMol_val)) in Traj:
                        NMol_temp = NMol_val
                param_value = "[{}]".format(NMol_temp)
                
        temp_CGModel = temp_CGModel.replace((str(param_name)+'_DUMMY'), str(param_value))
    
    # Assign the number of Gaussian in basis set 
    temp_CGModel = temp_CGModel.replace('NumberGaussians_DUMMY', str(NumberGauss))
    
    
    #if NumberGauss == 2:
    #    temp_CGModel = temp_CGModel.replace('Use2ndGaussian_DUMMY', 'True')
    #    temp_CGModel = temp_CGModel.replace('Use3rdGaussian_DUMMY', 'False')
    #elif NumberGauss == 3:
    #    temp_CGModel = temp_CGModel.replace('Use2ndGaussian_DUMMY', 'True')
    #    temp_CGModel = temp_CGModel.replace('Use3rdGaussian_DUMMY', 'True')
    #else:
    #    temp_CGModel = temp_CGModel.replace('Use2ndGaussian_DUMMY', 'False')
    #    temp_CGModel = temp_CGModel.replace('Use3rdGaussian_DUMMY', 'False')
    
    temp_CGModel = temp_CGModel.replace('CGMap_DUMMY', str(CGMap))
    temp_CGModel = temp_CGModel.replace('UseExpandedEnsemble_DUMMY', str(ExpEnsemble))
    
    if RunSpline:
        temp_CGModel = temp_CGModel.replace('RunSpline_DUMMY', 'True')
    else:
        temp_CGModel = temp_CGModel.replace('RunSpline_DUMMY', 'False')
    
    with open(CGModelScript, 'w') as g:
        g.write(temp_CGModel)
        
    # Submit Job
    call_1 = "qsub submit.sh"

    print(call_1)
    p1 = prcs.Popen(call_1, stdout=prcs.PIPE, shell=True)	
    (output, err) = p1.communicate()
    p_status = p1.wait()
    print("{}".format(err))

    with open("cgsweep_submit.log",'w') as logout:
        logout.write(output.decode("utf-8"))
    
    # Move backup one directory
    os.chdir("..")

''' ********************************************************************************* '''
''' ********* THE CODE THAT CALLS THE ABOVE FUNCTIONS TO GENERATE CG RUNS *********** '''
''' ********************************************************************************* '''

cwd = os.getcwd()
# Read in the cgmodel_sweep.py script.
# This is the script controlling the Srel Optimization.
with open(CGModelScript, 'r') as g:
    CGModel = g.read()

# Load in the submit script 
with open(str(SubmitScriptName), 'r') as g:
    temp_CGSubmitScript = g.read()

if ExpEnsemble == False: # For single-state point optimizations
    for i,Traj in enumerate(TrajList): # Loops through trajectory list
        for CGMap in CG_Mappings: # The monomer mapping ratio
            if RunSpline: 
                NumberGauss = 1
                RunDirName = str(Traj+'_CGMap_{}_Spline_{}'.format(CGMap,SpecialName))
                RunName = RunDirName
                # Create the CG directory
                CreateCGModelDirectory(ExpEnsemble, RunDirName,Traj,cwd,CGModel,CGModel_ParameterNames, CGModel_Parameters, 
                                            CGMap, RunSpline, NumberGauss, SubmitScriptName, temp_CGSubmitScript, NumberThreads, RunName, JobRunTime)
            if RunGauss:
                for NumberGauss in NumberGaussianBasisSets:
                    RunDirName = str(Traj+'_CGMap_{}_GaussBasis_{}_{}'.format(CGMap,NumberGauss,SpecialName))
                    RunName = RunDirName
                    temp_RunSpline = False
                    # Create the CG directory
                    CreateCGModelDirectory(ExpEnsemble, RunDirName,Traj,cwd,CGModel,CGModel_ParameterNames, CGModel_Parameters, 
                                                CGMap, temp_RunSpline, NumberGauss, SubmitScriptName, temp_CGSubmitScript, NumberThreads, RunName, JobRunTime)

elif ExpEnsemble == True:
    for i,Traj in enumerate(TrajList): # for ExpEnsemble, expects TrajList to be a list-of-list!
        for CGMap in CG_Mappings: # The monomer mapping ratio
            if RunSpline: 
                RunDirName = str('ExpEns_CGMap_{}_Spline_{}'.format(CGMap,SpecialName))
                RunName = RunDirName
                NumberGauss = 1
                # Create the CG directory
                CreateCGModelDirectory(ExpEnsemble, RunDirName,Traj,cwd,CGModel,CGModel_ParameterNames, CGModel_Parameters, 
                                            CGMap, RunSpline, NumberGauss, SubmitScriptName, temp_CGSubmitScript, NumberThreads, RunName, JobRunTime, TrajListInd = i)
            if RunGauss:
                for NumberGauss in NumberGaussianBasisSets:
                    RunDirName = str('ExpEns_CGMap_{}_GaussBasis_{}_{}'.format(CGMap,NumberGauss,SpecialName))
                    RunName = RunDirName
                    temp_RunSpline = False
                    # Create the CG directory
                    CreateCGModelDirectory(ExpEnsemble, RunDirName,Traj,cwd,CGModel,CGModel_ParameterNames, CGModel_Parameters, 
                                                CGMap, temp_RunSpline, NumberGauss, SubmitScriptName, temp_CGSubmitScript, NumberThreads, RunName, JobRunTime)

               

                   
    
        

