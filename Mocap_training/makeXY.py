import scipy.io
import os
import numpy as np
import tensorflow as tf
import itertools
from tensorflow.keras.utils import Sequence
import pdb;
from matplotlib import pyplot as plt
# ---------------------------------------------------------------------------- #
# CONSTANTS
PATH = 'E:/XYZ_Data/'
NUM_JOINTS = 32 #31 local coordinates + 1 global coordinate
NF = 12 # 0.2 seconds predicted
NB = 15 # 0.25 previous seconds considered
NUM_SUBJECTS = 112 #labels 1 to 143, some missing.
LAST_SUBJECT = 143 #143 actual, 2 for debug
TARGET_FPS = 60 #FPS we expect to process at. most mocap was recorded at 120
NUM_TRIALS = 2493 #total number of trials over all 112 subjects
NUM_TRIALS = 28 #trials in first 3 subjects for debugging
# ---------------------------------------------------------------------------- #
# Functions
# condense by slidign windows
# add input skipping N number of frames when building

# inside generators
    # load just N number of samples for each batch: a trial per thing is too big

def loadMocapFromMAT(path,subject,trial):
    if len('{}'.format(trial)) == 1:
        Ttag = '0{}'.format(trial)
    else:
        Ttag = '{}'.format(trial)
    if len('{}'.format(subject)) == 1:
        Stag = '0{}.mat'.format(subject)
    else:
        Stag = '{}.mat'.format(subject)
    filename = path + Stag
    if os.path.isfile(filename):
        try:
            temp = scipy.io.loadmat(filename)['SubjectData']['Data'+Ttag][0][0][0][0]
            ret = 1
            fps = temp[0][0][0]
            data = np.zeros( (len(temp[1][0]),NUM_JOINTS,3) )
            for i in range(0,len(temp[1][0])):
                data[i] = np.vstack([temp[2][i],temp[1][0][i]])
            return ret,data,fps
        except:
            return 0,0,0
    else:
        return 0,0,0

def getTrialXY(data,fps,batch_size = -1,batch_num = 0):
    # want output shape to be (samples,NF,32,3)
    # samples, frames from the 'future', 31 joints + 1 global coordinate, xyz
    ratio       = int(fps/TARGET_FPS)
    samples     = len(data) - ratio*NF - ratio*(NB-1)
    framesAgo   = ratio*(NB-1)
    framesAhead = ratio*NF
    if batch_size == -1: #whole thing
        start = framesAgo
        end = samples+framesAgo
        Y           = np.zeros((samples,NF,NUM_JOINTS,3))
        X           = np.zeros((samples,NB,NUM_JOINTS,3))
    else: # just one batch of a size
        start = batch_num*batch_size + framesAgo
        end = (batch_num+1)*batch_size -1 + framesAgo
        if end > samples+framesAgo: #end of the window not valid, less than batch_size X,Y produced
            end = samples+framesAgo
        if start > samples + framesAgo: #start of window not valid --> no XY
            return None,None
        Y           = np.zeros((end-start+1,NF,NUM_JOINTS,3))
        X           = np.zeros((end-start+1,NB,NUM_JOINTS,3))

    for i in range(start,end):
        for k in range(0,NB):
            X[i-start][k] = data[i-framesAgo+k*ratio]
        for k in range(0,NF):
            Y[i-start][k] = data[i+1+ratio*k]
#    pdb.set_trace()
    return X,Y

def getSubjectXY(subject):
    Done = 0
    trial = 1
    #pdb.set_trace()
    while not Done:
            ret,data,fps = loadMocapFromMAT(PATH,subject,trial)
            if ret:
                Xtemp,Ytemp  = getTrialXY(data,fps)
                try:
                    X = np.vstack([X,Xtemp])
                    Y = np.vstack([Y,Ytemp])
                except:
                    X = Xtemp
                    Y = Ytemp
                trial += 1
            else:
                Done = 1
    return X,Y

class MY_Generator(Sequence):
    def __init__(self,batch_size,subjects = range(0,LAST_SUBJECT),testing = False):
        # assigned batch_size, and list of subjects from database to generate from
        self.batch_size = batch_size
        self.subjects = subjects
        # load metadata file produced by output of countTrials
        trial_metadata = scipy.io.loadmat(PATH + 'trial_metadata')
        # length = length(subjects), number of trials in each
        self.num_trials =  [trial_metadata['num_trials'][0][i-1] for i in self.subjects]
        # shape = length(subject),num trials of that subject. (1D list of 1D arrays)
        self.fps_trials = [trial_metadata['fps_trials'][0][i-1] for i in self.subjects]
        # shape = length(subject),num trials of that subject. (1D list of 1D arrays)
        self.num_samples =  [trial_metadata['num_samples'][0][i-1] for i in self.subjects]
        # Initialize batch count, to be populated in __len__. same shape as num_samples
        self.num_batches = [0*trial_metadata['num_samples'][0][i-1] for i in self.subjects]
        self.total_batches = None
        # indexing values
        self.idx           = 0
        self.start_subject = self.subjects[self.idx]
        self.subject       = self.subjects[self.idx]
        self.trial         = 1
        self.batch         = 0

        #debug flag
        self.testing = testing
        #flag to avoid redundant data loading
        self.lastLoad = [None,None]
        #load first dataset
        self.loadData(message = "failed init")

    def __len__(self):
        if self.total_batches is None:
            self.total_batches = 0
            for i in range(0,len(self.subjects)):
                for j in range(0,self.num_trials[i]):
                    ratio = int( self.fps_trials[i][0][j] / TARGET_FPS)
                    reduced = (  self.num_samples[i][0][j] - ratio * (NF + NB -1) )
                    self.num_batches[i][0][j] = np.ceil( ( reduced )/self.batch_size )
                    self.total_batches  += self.num_batches[i][0][j]
        if self.testing:
            return 85
        return int(self.total_batches)

    def __getitem__(self,idx):
        X,Y = getTrialXY(self.data,self.fps,batch_size = self.batch_size, batch_num = self.batch)
        self.nextBatch()
        if self.testing:
            YY = 2*X[:,0:12,:,:]
            return X, YY
        return X,Y

    def nextSubject(self):
        #print('next subject \r')
        # reset trial
        self.trial = 1
        self.batch = 0
        # move subject
        self.idx += 1
        if self.idx >= len(self.subjects): # if idx impossible, must be a new training epoch
            self.subject = self.start_subject
            self.idx = 0
        else:                              # idx possible, just move on to next
            self.subject = self.subjects[self.idx]
            if self.num_trials[self.subjects-1] == 0: #skip empty subjects (like 4)
                self.nextSubject()
        self.loadData(message = "failed nextSubject")

    def nextTrial(self):
        #print('next trial \r')
        # reset batches
        self.batch = 0
        # move trial
        if self.testing:
            #print('\r',self.subject,self.trial,'\r')
            pass
        else:
            self.trial += 1


        # if exceed possible, next subject
        if self.trial > self.num_trials[self.subject-1]:
            self.nextSubject()
        else:
            self.loadData(message = "failed next Trial")

    def nextBatch(self):
        #print('next batch \r')
        self.batch += 1
        #pdb.set_trace()
        if self.batch >= self.num_batches[self.subject-1][0][self.trial-1]:
            self.nextTrial()

    def loadData(self,message = None):
        # load new data

        if ([self.subject,self.trial] == self.lastLoad):
            return
        self.ret,self.data, self.fps = loadMocapFromMAT(PATH,self.subject,self.trial)
        self.lastLoad = [self.subject,self.trial]
        # sanity check
        if not self.ret:
            if message is not None:
                print(message)
            print('no mocap loaded for subject {}: {}'.format(self.subject,self.trial))
            pdb.set_trace()

#used once now probably not needed

def countTrials(last_subject = LAST_SUBJECT):
    trialCount = 0
    subject = 1
    trial = 1
    numTrials = []
    fpsTrials = []
    numSamples = []
    last = 0
    while subject<= last_subject:
        #print('here')
        trial = 1
        samplecount = []
        thisFPS = []
        while True:
            #print('inner')
            ret,temp,fps = loadMocapFromMAT(PATH,subject,trial)
            if ret:
                print('s: {}  t: {}  c: {}'.format(subject,trial,trialCount+1))
                trialCount += 1
                samplecount.append(len(temp))
                thisFPS.append(fps)
                trial += 1
            else:
                numTrials.append(trialCount-last)
                numSamples.append(samplecount)
                fpsTrials.append(thisFPS)
                last = trialCount
                subject += 1
                break
    return numTrials,fpsTrials,numSamples
# def saveXYtoMAT(X,Y,subject,NB,NF,path = None):
#     XY = dict([('X',X),('Y',Y)])
#     if len('{}'.format(subject)) == 1:
#         Stag = '0{}'.format(subject)
#     else:
#         Stag = '{}'.format(subject)
#     filename = Stag+'_XY_{}_{}.mat'.format(NB,NF)
#     if path is not None:
#         filename = path + filename
#     #pdb.set_trace()
#     scipy.io.savemat(filename,XY,appendmat = False)
#
# def loadXYfromMAT(subject,NB,NF,path = None):
#     if len('{}'.format(subject)) == 1:
#         Stag = '0{}'.format(subject)
#     else:
#         Stag = '{}'.format(subject)
#     filename = Stag+'_XY_{}_{}.mat'.format(NB,NF)
#     if path is not None:
#         filename = path + filename
#     XY = scipy.io.loadmat(filename,appendmat = False)
#     X = XY['X']
#     Y = XY['Y']
#     return X,Y
