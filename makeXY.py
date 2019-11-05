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
    #pdb.set_trace()
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
        Y           = np.zeros((batch_size,NF,NUM_JOINTS,3))
        X           = np.zeros((batch_size,NB,NUM_JOINTS,3))
    if end > samples+framesAgo:
        return None,None
    for i in range(start,end):
        for k in range(0,NB):
            X[i-start][k] = data[i-framesAgo+k*ratio]
        for k in range(0,NF):
            Y[i-start][k] = data[i+1+ratio*k]
    pdb.set_trace()
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
        self.batch_size = batch_size
        self.subjects = subjects

        trial_metadata = scipy.io.loadmat(PATH + 'trial_metadata')
        self.num_trials =  [trial_metadata['num_trials'][0][i-1] for i in self.subjects]
        self.fps_trials = [trial_metadata['fps_trials'][0][i-1] for i in self.subjects]
        self.num_samples =  [trial_metadata['num_samples'][0][i-1] for i in self.subjects]

        self.idx = 0
        self.start_subject = self.subjects[self.idx]
        self.subject = self.subjects[self.idx]
        self.trial = 1
        self.ret,self.data, self.fps = loadMocapFromMAT(PATH,self.subject,self.trial)
        self.testing = testing
        #pdb.set_trace()

    def __len__(self):
        num_trials = 0
        num_XY = 0
        for i in range(0,len(self.subjects)):
            for j in range(0,self.num_trials[i]):
                #pdb.set_trace()
                ratio = int(self.fps_trials[i][0][j]/TARGET_FPS)
                num_XY += self.num_samples[i][0][j] - ratio*(NF + NB -1)
                #print(num_XY)
                num_trials += 1
            #print(num_XY)
        #pdb.set_trace()
        self.num_XY = num_XY
        return int(np.ceil(self.num_XY/float(self.batch_size)))

    def __getitem__(self,idx):



        if self.testing:
            YY = 2*X[:,0:12,:,:]
            #print(X.shape,YY.shape)
            #pdb.set_trace()
            #print(np.sum(X),np.sum(YY))
            return X,YY
        else:
            return X,Y

    def nextSubject(self):
        self.idx += 1
        if self.idx >= len(self.subjects): #new epoch
            self.subject = self.start_subject
            self.idx = 0
        else:
            self.subject = self.subjects[idx]
        self.trial = 1
        self.ret,self.data, self.fps = loadMocapFromMAT(PATH,self.subject,self.trial)
        if not self.ret:
            print('no mocap loaded for subject {}: {}'.format(self.subject,self.trial))
            pdb.set_trace()

    def nextTrial(self):
        self.trial += 1
        if self.trial > self.num_trials[self.subject-1]:
            self.nextSubject()
        else:
            self.ret,self.data, self.fps = loadMocapFromMAT(PATH,self.subject,self.trial)




#used once now probably not needed

# def countTrials(last_subject = LAST_SUBJECT):
#     trialCount = 0
#     subject = 1
#     trial = 1
#     numTrials = []
#     fpsTrials = []
#     numSamples = []
#     last = 0
#     while subject<= last_subject:
#         #print('here')
#         trial = 1
#         samplecount = []
#         thisFPS = []
#         while True:
#             #print('inner')
#             ret,temp,fps = loadMocapFromMAT(PATH,subject,trial)
#             if ret:
#                 print('s: {}  t: {}  c: {}'.format(subject,trial,trialCount+1))
#                 trialCount += 1
#                 samplecount.append(len(temp))
#                 thisFPS.append(fps)
#                 trial += 1
#             else:
#                 numTrials.append(trialCount-last)
#                 numSamples.append(samplecount)
#                 fpsTrials.append(thisFPS)
#                 last = trialCount
#                 subject += 1
#                 break
#     return numTrials,fpsTrials,numSamples
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
