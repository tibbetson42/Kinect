import scipy.io
import os
import numpy as np
import tensorflow as tf
import itertools
from tensorflow.keras.utils import Sequence
import pdb;
from matplotlib import pyplot as plt

import makeXY as cf
# ---------------------------------------------------------------------------- #

# Functions
# condense by slidign windows
# add input skipping N number of frames when building

# inside generators
    # load just N number of samples for each batch: a trial per thing is too big
#SCRIPT
# ---------------------------------------------------------------------------- #
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(40,activation = 'relu'),
    tf.keras.layers.Dense(40,activation = 'relu'),
    tf.keras.layers.Dense(cf.NF*cf.NUM_JOINTS*3),
    tf.keras.layers.Reshape((cf.NF,cf.NUM_JOINTS,3))
])


modeltest = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(40,activation = 'relu'),
    tf.keras.layers.Dense(40,activation = 'relu'),
    tf.keras.layers.Dense(cf.NF*cf.NUM_JOINTS*3),
    tf.keras.layers.Reshape((cf.NF,cf.NUM_JOINTS,3))
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

training_generator = cf.MY_Generator(batch_size = 32,subjects = [1], testing = True)
print('num of batches:  ',len(training_generator))
# trial_metadata = scipy.io.loadmat(PATH + 'trial_metadata')
# fps_trials = np.array(trial_metadata['fps_trials'][0])
# num_trials =  np.array(trial_metadata['num_trials'][0])
# num_samples =  np.array(trial_metadata['num_samples'][0])


# if testing, just do linear scaling see if can model
if training_generator.testing:
    re,de,fe = cf.loadMocapFromMAT(cf.PATH,1,1)
    XX,YY = cf.getTrialXY(de,fe)
    YY = 2*XX[:,0:12,:,:]
    filename = 'history_test'
else: #otherwise, look at comparison between fit and generator with subject 1
    XX,YY = cf.getSubjectXY(1)
    filename = 'history'
#
#pdb.set_trace()
historytest = modeltest.fit(XX,YY,epochs = 20)
history = model.fit_generator(generator = training_generator,epochs = 20)
#
plt.plot(historytest.history['loss'])
plt.plot(history.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['fit', 'gen'], loc='upper right')
plt.show()


results = dict([('generator',history.history['loss']),('fit',historytest.history['loss'])])
scipy.io.savemat(filename,results)
#
# re,de,fe = loadMocapFromMAT(PATH,5,1)
# XX,YY = getTrialXY(de,fe)
# Y = model.predict(XX)
# Ytest = modeltest.predict(XX)
# print(np.sum(XX),np.sum(Y),np.sum(Ytest))



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
