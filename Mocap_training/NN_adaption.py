
import pdb;
import os
import tensorflow as tf
import NN_functions as nnf
import Adaption_functions as af
import SKELETON_LAYOUT as sk
import DataManagement as dm



adaption_subject = 69

f = dm.filenames['dec8_validated'][0]
subs,jnts,layers,batches,fps = dm.filename2params(f)
checkpoint_path = "{}{}/cp.ckpt".format(nnf.MODEL_PATH,f)
checkpoint_dir = os.path.dirname(checkpoint_path)
model = tf.keras.models.load_model(checkpoint_dir + '/model_' + f +'.h5')

generator = nnf.MY_Generator(batch_size = 1,subjects = [adaption_subject],joints = jnts,target_fps = fps)
X0 = generator[0][0]

adapter = af.Weight_Adapter(model,X0)
adapter.start(X0)
for i in range(0,10):
    adapter.feedData(generator[0][0])
    adapter.mainLoop()
    #need to have something for the X value out of the generator
