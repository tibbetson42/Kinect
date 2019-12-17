import scipy.io
import os
import numpy as np
import tensorflow as tf
import itertools
import pdb;
from matplotlib import pyplot as plt
import datetime
import NN_functions as cf
import SKELETON_LAYOUT as SK
import DataManagement as dm


class State_Comparer():
    def __init__(self,model,test_subject, fps = cf.TARGET_FPS):
        self.generator = cf.MY_Generator(batch_size = 1, subjects = [test_subject], testing = True, joints = jnts, target_fps = fps)
        self.model = model
        self.subject = test_subject
    def __len__(self):
        return len(self.generator)
    def predict_trial(self,trial = 1):
        if trial > self.generator.num_trials[0]:
            print('warning, trial {} does not exist for sub {}. Showing trial 1 instead'.format(trial,self.subject))
            trial = 1
        Ystates_t,Ystates_e,Xstates = None,None,None
        times = None
        i = 0
        while i<self.generator.num_batches[0][0][trial-1]:
            self.fps = float(self.generator.fps)
            X,Ytrue = self.generator[0]
            X *= 1/0.45; Ytrue *= 1/0.45
            Yeval = self.model.predict(X)
            err = np.zeros((1,X.shape[2]))
            #pdb.set_trace()
            for j in range(0,X.shape[2]):
                err[0][j] = np.array([ np.linalg.norm(Ytrue[0][-1][j] - Yeval[0][-1][j]) ])

            if Ystates_t is None and times is None:
                times = np.array([0])
                Ystates_t = self.getState(Ytrue[0])
                Ystates_e = self.getState(Yeval[0])
                Xstates   = self.getState(X[0])
                errstates = np.array(err)
                #pdb.set_trace()
            else:
                times = np.vstack([times,[times[-1]+1/self.fps]])
                Ystates_t = np.vstack([Ystates_t,self.getState(Ytrue[0])])
                Ystates_e = np.vstack([Ystates_e,self.getState(Yeval[0])])
                Xstates = np.vstack([Xstates,self.getState(X[0])])
                errstates = np.vstack([errstates,err])
                #pdb.set_trace()
            i+=1
        return times,Xstates,Ystates_t,Ystates_e,errstates

    def predict_subject(self):
        times,Xstates,Ystates_t,Ystates_e,errstates = self.predict_trial(trial = 1)
        for trial in range(2,self.generator.num_trials[0] + 1):
            print('evaluating trial {}'.format(trial))
            t,Xstates,Ystates_t,Ystates_e,e = comparer.predict_trial(trial = trial)
            times = np.vstack([times, t + times[-1]])
            Xstates = np.vstack([Xstates,Xstates])
            Ystates_t = np.vstack([Ystates_t,Ystates_t])
            Ystates_e = np.vstack([Ystates_e,Ystates_e])
            errstates = np.vstack([errstates,e])
        return times,Xstates,Ystates_t,Ystates_e,errstates

    def getState(self,X):
        state = X[-1].reshape((1,-1))
        return state


f = dm.filenames['dec7_retrain'][0]
f = dm.filenames['dec8_validated'][0]
subs,jnts,layers,batches,fps = dm.filename2params(f)
checkpoint_path = "{}/cp.ckpt".format(f)
checkpoint_dir = os.path.dirname(checkpoint_path)
model = tf.keras.models.load_model(checkpoint_dir + '/model_' + f +'.h5')

test_subject = 35
comparer = State_Comparer(model,test_subject, fps = fps)
times,X,Yt,Ye,err = comparer.predict_trial(trial = 1)
fig, axes = plt.subplots(1,len(jnts)+1)

for i in range(0,len(jnts)):
    #axes[i].plot(times,X[:,3*i:3*(i+1)], 'g')
    axes[i].plot(times,Yt[:,3*i:3*(i+1)],'b')
    axes[i].plot(times,Ye[:,3*i:3*(i+1)],'r')
    axes[i].grid()
    axes[i].legend(['Yt',None,None,'Ye'],loc = 'best')
    axes[-1].plot(times,err[:,i],label = 'L2 error joint {}'.format(jnts[i]))

#batch_size 1
#batch num 0 start = 4; end = 11 [0,2,4] vs [6,8,10]
#batch num 1 start = 5; end = 12 [1,3,5] vs [7,9,11]

#batch_size 2
#batch num 0 start = 4; end = 12 [0,2,4],[1,3,5] vs [6,8,10],[7,9,11]
#batch num 1 start = 6; end = 14 [2,4,6],[3,5,7] vs [8,10,12],[9,11,13]

#batch_size 3
#batch num 0 start = 4; end = 13 [0,2,4],[1,3,5],[2,4,6] vs [6,8,10],[7,9,11],[8,10,12]
#batch num 1 start = 7; end = 16 [3,5,7],[4,6,8],[5,7,9] vs [9,11,13],[10,12,14],[11,13,15]
