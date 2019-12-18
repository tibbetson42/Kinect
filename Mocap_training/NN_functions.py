import time
import os
import itertools

import numpy                as np
import scipy.io
from scipy                  import linalg
import tensorflow           as tf
from tensorflow.keras.utils import Sequence

from matplotlib             import pyplot as plt

from threading 				import Thread

import SKELETON_LAYOUT      as sk
import DataManagement       as dm
import pdb;

# ---------------------------------------------------------------------------- #
# CONSTANTS
PATH = 'E:/XYZ_Data/'
MODEL_PATH = 'Models/'
#PATH = 'XYZ_Data/'
NUM_JOINTS = 31 #31 local coordinates + 1 global coordinate
NF = 3 # 0.2 seconds predicted
NB = 3 # 0.25 previous seconds considered
NUM_SUBJECTS = 112 #labels 1 to 143, some missing.
LAST_SUBJECT = 143 #143 actual, 2 for debug
TARGET_FPS = 60 #FPS we expect to process at. most mocap was recorded at 120
# ------------------------------------------------------------------------- #
# Functions
# condense by slidign windows
# add input skipping N number of frames when building

def create_model(layer1 = 40, layer2 = 40,num_joints = 31,for_adapter = False):
    #Create Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer( input_shape = ( NB,num_joints,3) ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(40,activation = 'relu'),
        tf.keras.layers.Dense(40,activation = 'relu')
    ])
    if not for_adapter:
        model.add(   tf.keras.layers.Dense( NF*num_joints*3 )   )
        model.add(   tf.keras.layers.Reshape( (NF,num_joints,3) )  )

    from tensorflow.keras.optimizers import Adam
    model.compile(
        loss = 'mean_squared_error',
        optimizer = Adam(lr = 0.001)
    )
    return model

def loadMocapFromMAT(path,subject,trial,joints = sk.FULL_BODY):
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
        #try:
            temp = scipy.io.loadmat(filename)['SubjectData']['Data'+Ttag][0][0][0][0]
            ret = 1
            fps = temp[0][0][0]
            data = np.zeros( (len(temp[1][0]),len(joints),3) )
            if (joints == sk.HIP_CENTER).any():
                for i in range(0,len(temp[1][0])):
                    data[i] = np.vstack([temp[2][i],temp[1][0][i][joints]])
            else:
                for i in range(0,len(temp[1][0])):
                    data[i] = np.array(temp[1][0][i][joints])
            return ret,data,fps
    else:
        return 0,0,0

def getTrialXY(data,fps,batch_size = -1,batch_num = 0,target_fps = TARGET_FPS):
    # want output shape to be (samples,NF,32,3)
    # samples, frames from the 'future', 31 joints + 1 global coordinate, xyz
    ratio       = int(fps/target_fps)
    samples     = len(data) - ratio*NF - ratio*(NB-1)
    framesAgo   = ratio*(NB-1)
    framesAhead = ratio*NF
    if batch_size == -1: #whole thing
        batch_size = samples
        start = framesAgo
        end = samples+framesAgo
        Y           = np.zeros((samples,NF,len(data[0]),3))
        X           = np.zeros((samples,NB,len(data[0]),3))
    else: # just one batch of a size
        max_batch = int(samples/batch_size)
        start = batch_num*batch_size + framesAgo
        end = start+NF+NB+batch_size
        if end >= len(data): #end of the window not valid, less than batch_size X,Y produced
            end = len(data)-1
        if batch_num > max_batch: #start of window not valid --> no XY
            print('getTrialXY failure')
            print('batch number error')
            print('data, with NB {}, NF {}, batch size {} only has {} batches'.format(NB,NF,batch_size,max_batch))
            pdb.set_trace()
            return None,None
        Y           = np.zeros((batch_size,NF,len(data[0]),3))
        X           = np.zeros((batch_size,NB,len(data[0]),3))

    #pdb.set_trace()
    for i in range(0,batch_size):
        for k in range(0,NB):

            idx = start-framesAgo+k*ratio
            #print(idx)
            X[i][k] = data[idx]
        for k in range(0,NF):
            idx = start + (k+1)*ratio
            #print(idx)
            Y[i][k] = data[idx]
    #pdb.set_trace()
    return X,Y

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class MY_Generator(Sequence):
    def __init__(self,batch_size,subjects,threading = False,joints = sk.FULL_BODY,target_fps = TARGET_FPS,randomize_direction = False,suppress = False):
        self.joints = joints
        self.target_fps = target_fps

        # assigned batch_size, and list of subjects from database to generate from
        self.batch_size = batch_size

        # load metadata file produced by output of countTrials
        trial_metadata = scipy.io.loadmat(PATH + 'trial_metadata')
        # length = length(subjects), number of trials in each
        num_trials =  [trial_metadata['num_trials'][0][i-1] for i in subjects]

        #remove empty subjects
        self.subjects = [subjects[x] for i,x in np.ndenumerate(np.nonzero(num_trials))]
        self.num_trials = [num_trials[x] for i,x in np.ndenumerate(np.nonzero(num_trials))]

        #load rest of metadata
        # shape = length(subject),num trials of that subject. (1D list of 1D arrays)
        self.fps_trials = [trial_metadata['fps_trials'][0][i-1] for i in self.subjects]
        # shape = length(subject),num trials of that subject. (1D list of 1D arrays)
        self.num_samples =  [trial_metadata['num_samples'][0][i-1] for i in self.subjects]
        # Initialize batch count, to be populated in __len__. same shape as num_samples
        self.num_batches = [0*trial_metadata['num_samples'][0][i-1] for i in self.subjects]
        self.total_batches = None

        # indexing values
        self.idx           = 0
        self.subject       = self.subjects[self.idx]
        self.trial         = 1
        self.batch         = 0

        #debug flag
        self.threading = threading
        #flag to avoid redundant data loading
        self.lastLoad = [None,None]
        # flag to randomize direction of X so that trainer is not biased
        # if data was largely collected facing a particular direction
        self.randomize_direction = randomize_direction

        #load first dataset
        if self.threading:
            self.Loader = DataLoader(self)
            self.Loader.start()
            while not self.Loader.ready:
                time.sleep(0.01)

        self.loadData(message = '\ninit fail\n\n')
        if not suppress:
            print('\ngenerator init succesful')
            print('\nnum of batches:  ',len(self),' of size {}'.format(self.batch_size))
            print('num of subjects: ',len(self.subjects))
            print('subjects: {}'.format(dm.array2str(self.subjects,'')))
            print('{} joints to train\n{} past frames predict\n{} future frames at \n{} fps'.format(len(self.joints),NB,NF,self.target_fps))
            print('each frame has shape {}'.format(self.data.shape[1::]))

    def __len__(self):
        if self.total_batches is None:
            self.total_batches = 0
            for i in range(0,len(self.subjects)):
                for j in range(0,self.num_trials[i]):
                    ratio = int( self.fps_trials[i][0][j] / self.target_fps)
                    reduced = (  self.num_samples[i][0][j] - ratio * (NF + NB -1) )
                    self.num_batches[i][0][j] = np.ceil( ( reduced )/self.batch_size )
                    self.total_batches  += self.num_batches[i][0][j]
        return int(self.total_batches)

    def __getitem__(self,idx):
        X,Y = getTrialXY(self.data,self.fps,batch_size = self.batch_size, batch_num = self.batch,target_fps = self.target_fps)
        if X is None:
            pdb.set_trace()
        if self.randomize_direction:
            #X,Y = self.rotateXY(X,Y,perbatch = False)
            X,Y = self.rotateXY(X,Y)

        self.nextBatch()
        return X,Y

    def rotateXY(self,X,Y,perbatch = True):
        ang = np.random.rand()*2*np.pi
        c = np.cos(ang); s = np.sin(ang)
        R = np.array([[c,s,0],[-s,c,0],[0,0,1]])
        for i in range(0,len(X)):
            if not perbatch:
                ang = np.random.rand()*2*np.pi
                c = np.cos(ang); s = np.sin(ang)
                R = np.array([[c,s,0],[-s,c,0],[0,0,1]])
            for j in range(0,len(X[i])):
                X[i][j,:,:] = np.matmul(X[i][j,:,:],R)
            for j in range(0,len(Y[i])):
                Y[i][j,:,:] = np.matmul(Y[i][j,:,:],R)
        return X,Y

    def reset(self,Subject = None):
        if Subject is None:
            self.idx = 0
        else:
            try:
                self.idx = self.subjects.index(Subject)
            except:
                print("ERROR: tried to reset generator to a subject ({}) outside its domain. Check initialization of generator".format(Subject))
                print("manually reset/ save anything using pdb now")
                pdb.set_trace()

        self.subject       = self.subjects[self.idx]
        self.trial         = 1
        self.batch         = 0
        self.updateLoader()
        return

    def updateLoader(self):
        self.Loader.trial = self.trial + 1
        if self.Loader.trial > self.num_trials[self.idx]:
            self.Loader.trial = 1
            self.Loader.idx = (self.idx + 1)%len(self.subjects)
        self.Loader.ready = False

    def loadData(self,message = None):
        # load new data
        if self.threading:
            self.ret,self.data, self.fps = self.Loader.getData()
            self.updateLoader()
        else:
            if ([self.subject,self.trial] == self.lastLoad):
                return
            self.ret, self.data, self.fps= loadMocapFromMAT(PATH,self.subjects[self.idx],self.trial,self.joints)
            self.lastLoad = [self.subject,self.trial]

        # sanity check
        if not self.ret:
            if message is not None:
                print(message)
            print('no mocap loaded for subject {}: {}'.format(self.subject,self.trial))
            pdb.set_trace()

    def nextSubject(self):
        #print('next subject')
        # reset trial
        self.trial = 1
        self.batch = 0
        # move subject
        self.idx  = (self.idx + 1)%len(self.subjects)                        # idx possible, just move on to next
        self.subject = self.subjects[self.idx]
        if self.num_trials[self.idx] == 0: #skip empty subjects (like 4)
            self.nextSubject()
        self.loadData(message = "failed nextSubject")

    def nextTrial(self):
        # reset batches
        self.batch = 0
        self.trial += 1

        # if exceed possible, next subject
        if self.trial > self.num_trials[self.idx]:
            self.nextSubject()
        else:
            self.loadData(message = "failed next Trial")

    def nextBatch(self):
        self.batch += 1
        if self.batch >= self.num_batches[self.idx][0][self.trial-1]:
            self.nextTrial()

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class DataLoader():
    def __init__(self,generator):
        self.subjects = generator.subjects
        self.num_trials = generator.num_trials
        self.idx = generator.idx
        self.trial = generator.trial
        self.joints = generator.joints
        self.stopped = True
        self.lastLoad = [None,None]
        self.thread =  Thread(target=self.update, args=(),daemon = True)
        self.ready = False

    def start(self):
        self.stopped = False
        self.thread.start()

    def stop(self):
        self.stopped = True

    def update(self):
        print('loading thread starting')
        while True:
            if self.stopped:
                break
            self.loadData()

    def loadData(self,message = None):
        if ([self.idx,self.trial] == self.lastLoad):
            time.sleep(0.05)
            return
        self.ret, self.data, self.fps= loadMocapFromMAT(PATH,self.subjects[self.idx],self.trial,joints = self.joints)
        self.ready = True
        #pdb.set_trace()
        self.lastLoad = [self.idx,self.trial]

    def getData(self):
        while not self.ready:
            time.sleep(0.1)
        return self.ret,self.data,self.fps
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Weight_Adapter():
    def __init__(self,model,sampleX):

        weights = model.get_weights();
        # Initialize per cheng paper on adaptive weights
        num_joints = sampleX[0].shape[-2]
        layers = weights[0].shape[1]
        # Offline trained model - everything but last layer
        self.U = [weights[i] for i in range(0,len(weights)-2)]
        self.g = create_model(layer1 =layers, layer2 = layers,num_joints = num_joints,for_adapter = True)
        # also pass one sample to it to initialize
        #pdb.set_trace()
        g_out = self.g.predict(sampleX)
        self.g.set_weights(self.U)

        #Format adaptive weights. W is in NN format, Theta is column stack of W
        self.W = weights[-2::]
        self.Theta = self.W2Theta(self.W)

        # Initialize state variables. X = past window, true value
        self.Xest = None
        self.X = None
        self.Xerr = None
        self.Xshape = sampleX.shape
        self.Glength = len(g_out) + 1

        # Adaption Gain
        self.F = 100 * np.eye(len(self.Theta))

        # Learning parameters
        self.Lambda1 = 0.998
        self.Lambda2 = 1.0

        return

    def start(self,X):
        print('starting...')
        self.X = X
        self.Phi   = self.getPhi( self.g.predict(self.X) )
        self.Xest = np.matmul( self.Phi, self.Theta ).reshape(self.Xshape)
        # new data flag
        self.new = False
        #pdb.set_trace()
        return

    def feedData(self,X):
        print('feeding data')
        self.X = X
        self.new = True
        return

    def mainLoop(self):
        print('mainLoop start')
        #pdb.set_trace()
        if not self.new:
            return
        self.Xerr  = (self.X - self.Xest)
        self.Theta = self.updateTheta( self.Theta, self.F, self.Phi, self.Xerr.reshape((-1,1)))
        self.F     = self.updateF( self.F, self.Phi )
        self.Phi   = self.getPhi( self.g.predict(self.X) )
        self.Xest  = np.matmul( self.Phi, self.Theta ).reshape(self.Xshape)
        self.new = False
        print('mainLoop end')
        pdb.set_trace()
        return

    def getPhi(self,g_out):
        g_out = np.append(g_out.reshape((1,-1)),1)
        Phi = g_out
        for i in range(0,self.W[0].shape[1]-1):
            Phi = linalg.block_diag(Phi,g_out)
        return Phi

    def updateF(self,F,Phi):
        #equation 7 in cheng's paper
        L1 = self.Lambda1
        L2 = self.Lambda2

        # Simplfication of writing
        PhiT = Phi.transpose()

        # Denominator as a matrix
        scaling = np.linalg.inv(L1 + L2 * np.linalg.multi_dot( [ Phi, F, PhiT ]  ))

        # Scale denominator (9x9 if X is 9x1) to match F (369x369 if 40 hidden layer nodes)
        dg = np.diag(scaling)
        Sdiag = np.zeros(Phi.shape[1])
        for si, s in enumerate(dg):
            Sdiag[si:si + self.Glength] = s
        S = np.diag(Sdiag)

        # Evaluate
        F_plus = 1/L1 * (F  - L2 * np.linalg.multi_dot([F,PhiT,Phi,F,S]))
        return F_plus

    def updateTheta(self,Theta,F,Phi,error):
        Theta_plus = self.Theta + np.linalg.multi_dot([F,Phi.transpose(),error])
        return Theta_plus



    def W2Theta(self, W):
        Theta = None
        for i in range(0,len(W[1])):
            ti = np.vstack([W[0][:,i].reshape((-1,1)),W[1][i]])
            if Theta is None:
                Theta = ti
            else:
                Theta = np.vstack([Theta,ti])
        return Theta
