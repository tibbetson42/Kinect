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


f = dm.filenames['dec8_validated'][0]
subs,jnts,layers,batches,fps = dm.filename2params(f)
checkpoint_path = "{}/cp.ckpt".format(f)
checkpoint_dir = os.path.dirname(checkpoint_path)
model = tf.keras.models.load_model(checkpoint_dir + '/model_' + f +'.h5')

adapter = cf.Weight_Adapter(model,jnts,layers,fps)
