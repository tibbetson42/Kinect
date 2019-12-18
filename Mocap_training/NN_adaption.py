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



adaption_subject = 69

f = dm.filenames['dec8_validated'][0]
subs,jnts,layers,batches,fps = dm.filename2params(f)
checkpoint_path = "{}{}/cp.ckpt".format(cf.MODEL_PATH,f)
checkpoint_dir = os.path.dirname(checkpoint_path)
model = tf.keras.models.load_model(checkpoint_dir + '/model_' + f +'.h5')

generator = cf.MY_Generator(batch_size = 1,subjects = [adaption_subject],joints = jnts,target_fps = fps)
X0 = generator[0][0]

adapter = cf.Weight_Adapter(model,X0)
adapter.start(X0)
for i in range(0,10):
    adapter.feedData(generator[0][0])
    adapter.mainLoop()
    #need to have something for the X value out of the generator
