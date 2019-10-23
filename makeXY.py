import scipy.io
import numpy as np
import tensorflow as tf
import itertools
from tensorflow.keras.utils import Sequence
import pdb;
# ---------------------------------------------------------------------------- #
# CONSTANTS
PATH = 'XYZ_Data/'
NUM_JOINTS = 32 #31 local coordinates + 1 global coordinate
NF = 12 # 0.2 seconds predicted
NB = 15 # 0.25 previous seconds considered
LAST_SUBJECT = 2 #143 actual, 2 for debug
TARGET_FPS = 60 #FPS we expect to process at. most mocap was recorded at 120

# ---------------------------------------------------------------------------- #
# Functions
# condense by slidign windows
# add input skipping N number of frames when building


# inside generators
    # load xyz for a batch (subcet)



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
    temp = scipy.io.loadmat(filename)['SubjectData']['Data'+Ttag][0][0][0][0]
    fps = temp[0][0][0]
    data = np.zeros( (len(temp[1][0]),NUM_JOINTS,3) )
    for i in range(0,len(temp[1][0])):
        data[i] = np.vstack([temp[2][i],temp[1][0][i]])
    return data,fps

def saveXYtoMAT(X,Y,subject,NB,NF,path = None):
    XY = dict([('X',X),('Y',Y)])
    if len('{}'.format(subject)) == 1:
        Stag = '0{}'.format(subject)
    else:
        Stag = '{}'.format(subject)
    filename = Stag+'_XY_{}_{}.mat'.format(NB,NF)
    if path is not None:
        filename = path + filename
    pdb.set_trace()
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

def getTrialXY(data,fps):
    # want output shape to be (samples,NF,32,3)
    # samples, frames from the 'future', 31 joints + 1 global coordinate, xyz
    ratio       = int(fps/TARGET_FPS)
    samples     = len(xyz) - ratio*NF - ratio*(NB-1)
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
        #try:
            xyzData,fps = loadMocapFromMAT(PATH,subject,trial)
            Xtemp,Ytemp  = getTrialXY(xyzData,fps)
            try:
                X = np.vstack([X,Xtemp])
                Y = np.vstack([Y,Ytemp])
            except:
                X = Xtemp
                Y = Ytemp
            trial += 1
        #except:
            Done = 1
            #print(trial)
    return X,Y


class MY_Generator(Sequence):
    def __init__(self,batch_size,start_subject = 1,num_subjects = 112):
        self.batch_size = batch_size
        self.num_subjects = num_subjects
        self.subject = start_subject
        self.trial = 1
        self.data, self.fps = loadMocapFromMAT(PATH,self.subject,self.trial)

    def __len__(self):
        return int(np.ceil(self.num_subjects/float(self.batch_size)))

    def __getitem__(self,idx):
        i = 0
        while i < self.batch_size:
            try:
                Xtemp,Ytemp = getSubjectXY(self.subject)
                print('subject {} loaded'.format(self.subject))
                i += 1
                #print(self.subject)
            except:
                pass
            try:
                X = np.vstack([X,Xtemp])
                Y = np.vstack([Y,Ytemp])
            except:
                X = Xtemp
                Y = Ytemp
            print('{} samples'.format(len(X)))
            pdb.set_trace()
            self.subject += 1
        return X,Y

    def nextSubject(self):
        self.Subject += 1
        self.Trial = 1
        self.data, self.fps = loadMocapFromMAT(PATH,self.subject,self.trial)

    def nextTrial(self):
        self.Trial += 1

# ---------------------------------------------------------------------------- #
model = tf.keras.models.Sequential([
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
    metrics = ['acc']
)
# history = model.fit(Xtrain,Ytrain,epochs = 5)
#pdb.set_trace()
[X,Y] = getSubjectXY(89)

training_generator = MY_Generator(1)
#print(training_generator.__len__)
pdb.set_trace()
model.fit_generator(generator = training_generator,epochs = 1)

# num_epochs = 1
# epoch = 1
# while epoch <= num_epochs:
Done = 0
subject = 1
#print('Epoch {}'.format(epoch))
# while not Done:
#     if subject > LAST_SUBJECT:
#         Done = 1
#         break
#     else: #subject and epoch = 1 then fit, else train_on_batch
#         print('     subject {} saving'.format(subject))
#         X,Y = getSubjectXY(subject)
#         saveXYtoMAT(X,Y,subject,NB,NF,PATH)
#         # pdb.set_trace()
#         # model.fit(Xtrain,Ytrain)
#         print('     subject {} finished'.format(subject))
#         subject += 1
    # else:
    #     try:
    #         print('     subject {} training'.format(subject))
    #         X,Y = getSubjectXY(subject)
    #         #pdb.set_trace()
    #         history = model.train_on_batch(X,Y)
    #         print('     subject {} finished'.format(subject))
    #         subject+=1
    #     except:
    #         print('     error: subject {} data not existing'.format(subject))
    #         subject += 1
    # epoch += 1









#test = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
