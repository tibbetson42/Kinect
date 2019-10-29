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

def getTrialXY(data,fps):
    # want output shape to be (samples,NF,32,3)
    # samples, frames from the 'future', 31 joints + 1 global coordinate, xyz
    #pdb.set_trace()
    ratio       = int(fps/TARGET_FPS)
    samples     = len(data) - ratio*NF - ratio*(NB-1)
    framesAgo   = ratio*(NB-1)
    framesAhead = ratio*NF
    Y           = np.zeros((samples,NF,NUM_JOINTS,3))
    X           = np.zeros((samples,NB,NUM_JOINTS,3))
    for i in range(framesAgo,samples+framesAgo):
        for k in range(0,NB):
            X[i-framesAgo][k] = data[i-framesAgo+k*ratio]
        for k in range(0,NF):
            Y[i-framesAgo][k] = data[i+1+ratio*k]
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
    def __init__(self,batch_size,start_subject = 1,num_trials = NUM_TRIALS,num_subjects = 112,testing = False):
        self.batch_size = batch_size
        self.num_subjects = num_subjects
        self.num_trials = num_trials

        self.start_subject = start_subject
        self.subject = start_subject
        self.trial = 1
        self.count = 0
        self.ret,self.data, self.fps = loadMocapFromMAT(PATH,self.subject,self.trial)
        self.trial_list = scipy.io.loadmat(PATH + 'trial_list')['trials'][0]
        self.testing = testing
        #pdb.set_trace()

    def __len__(self):
        return int(np.ceil(self.num_trials/float(self.batch_size)))

    def __getitem__(self,idx):
        #print('\rbatch idx:',idx)
        for i in range(0,self.batch_size):
            #print(i,self.subject,self.trial,len(self.data))
            Xtemp, Ytemp = getTrialXY(self.data,self.fps)
            try:
                X = np.vstack([X,Xtemp])
                Y = np.vstack([Y,Ytemp])
            except:
                X = Xtemp
                Y = Ytemp
            self.nextTrial()

        if self.testing:
            YY = 2*X[:,0:12,:,:]
            #print(X.shape,YY.shape)
            #pdb.set_trace()
            #print(np.sum(X),np.sum(YY))
            return X,YY
        else:
            return X,Y

    def nextSubject(self):
        self.subject += 1
        if self.subject >= self.start_subject + self.num_subjects: #new epoch
            self.subject = self.start_subject
        self.trial = 1
        self.ret,self.data, self.fps = loadMocapFromMAT(PATH,self.subject,self.trial)
        if not self.ret:
            print('no mocap loaded for subject {}: {}'.format(self.subject,self.trial))
            pdb.set_trace()

    def nextTrial(self):
        self.trial += 1
        self.count += 1
        if self.count >= len(self): #if number completed trials > number of trials then its a new epoch
            self.subject = self.start_subject
            self.trial = 1
            self.count = 0
            return
        self.ret,self.data, self.fps = loadMocapFromMAT(PATH,self.subject,self.trial)
        if self.ret:
            return 1
        else:
            #print('\rno more trials in subject {}'.format(self.subject))
            self.nextSubject()
            return 0





#SCRIPT
# ---------------------------------------------------------------------------- #
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(40,activation = 'relu'),
    tf.keras.layers.Dense(40,activation = 'relu'),
    tf.keras.layers.Dense(NF*NUM_JOINTS*3),
    tf.keras.layers.Reshape((NF,NUM_JOINTS,3))
])


modeltest = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(40,activation = 'relu'),
    tf.keras.layers.Dense(40,activation = 'relu'),
    tf.keras.layers.Dense(NF*NUM_JOINTS*3),
    tf.keras.layers.Reshape((NF,NUM_JOINTS,3))
])

from tensorflow.keras.optimizers import Adam
model.compile(
    loss = 'mean_squared_error',
    optimizer = Adam(lr = 0.001),
)
modeltest.compile(
    loss = 'mean_squared_error',
    optimizer = Adam(lr = 0.001),
)

training_generator = MY_Generator(start_subject = 1,batch_size = 1,num_subjects = 1,num_trials = 1,testing = True)


# TESTING for MODEL.FIT
re,de,fe = loadMocapFromMAT(PATH,1,1)
XX,YY = getTrialXY(de,fe)
YY = 2*XX[:,0:12,:,:]

historytest = modeltest.fit(XX,YY,epochs = 50,batch_size = 2699)
history = model.fit_generator(generator = training_generator,epochs = 50)

plt.plot(historytest.history['loss'])
plt.plot(history.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['fit', 'gen'], loc='upper right')

re,de,fe = loadMocapFromMAT(PATH,5,1)
XX,YY = getTrialXY(de,fe)
Y = model.predict(XX)
Ytest = modeltest.predict(XX)
print(np.sum(XX),np.sum(Y),np.sum(Ytest))
pdb.set_trace()









#used once now probably not needed
def countTrials():
    count = 0
    subject = 1
    trial = 1
    nums = []
    last = 0
    while subject<= LAST_SUBJECT:
        ret,temp,fps = loadMocapFromMAT(PATH,subject,trial)
        X,Y= getTrialXY(temp,fps)
        if ret:
            print('s: {}  t: {}  c: {}'.format(subject,trial,count+1))
            count += 1
            trial += 1
        else:
            nums.append(count-last)
            last = count
            print(nums)
            subject += 1
            trial = 1
    return nums

def countSamples()

def saveXYtoMAT(X,Y,subject,NB,NF,path = None):
    XY = dict([('X',X),('Y',Y)])
    if len('{}'.format(subject)) == 1:
        Stag = '0{}'.format(subject)
    else:
        Stag = '{}'.format(subject)
    filename = Stag+'_XY_{}_{}.mat'.format(NB,NF)
    if path is not None:
        filename = path + filename
    #pdb.set_trace()
    scipy.io.savemat(filename,XY,appendmat = False)

def loadXYfromMAT(subject,NB,NF,path = None):
    if len('{}'.format(subject)) == 1:
        Stag = '0{}'.format(subject)
    else:
        Stag = '{}'.format(subject)
    filename = Stag+'_XY_{}_{}.mat'.format(NB,NF)
    if path is not None:
        filename = path + filename
    XY = scipy.io.loadmat(filename,appendmat = False)
    X = XY['X']
    Y = XY['Y']
    return X,Y
