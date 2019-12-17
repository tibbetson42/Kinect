import scipy.io
import os
import numpy           as np
import tensorflow      as tf
import itertools
import pdb;
import datetime
import NN_functions    as cf
import SKELETON_LAYOUT as sk
import DataManagement  as dm
# ---------------------------------------------------------------------------- #

# Functions
# condense by slidign windows
# add input skipping N number of frames when building

# inside generators
    # load just N number of samples for each batch: a trial per thing is too big
#SCRIPT
# ---------------------------------------------------------------------------- #
#thread the generators? Increases speed about 2x
threading = True

# Subjects for Training and Validation (do we validate?)
training_subjects = np.hstack([range(1,25)])
validation_subjects = np.hstack([range(35,48)])
validation = True

# Training Params
target_fps = 30
epochs = 50
batch_sizes = [512] #batch_sizes = [128,256,512,1024,2048]
layer_sizes = [40,100] #layer_sizes = [40,100,cf.NB*cf.NUM_JOINTS,cf.NB*cf.NUM_JOINTS*3,cf.NB*cf.NUM_JOINTS*9]
joint_sets =  [sk.ARM_LEFT,np.array([sk.WRIST_LEFT])]
label = 'dec17_randomized'

#Init
files = []

#Train a model for each of the different parameter sets
for batch_size in batch_sizes:
    for layer_size in layer_sizes:
        for joints in joint_sets:
            filetag = dm.params2filename(label,training_subjects,joints,layer_size,batch_size,target_fps)
            files.append(filetag)
            checkpoint_path = "{}/cp.ckpt".format(filetag)
            #pdb.set_trace()
            checkpoint_dir = os.path.dirname(checkpoint_path)

            training_generator = cf.MY_Generator(batch_size = batch_size, subjects = training_subjects, target_fps = target_fps, threading = threading, joints = joints, randomize_direction = True)
            if validation:
                validation_generator = cf.MY_Generator(batch_size = batch_size, subjects = validation_subjects, target_fps = target_fps, threading = threading, joints = joints, randomize_direction = True )


            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)

            model = cf.create_model(layer1 = layer_size,layer2 = layer_size,joints = joints)

            if os.path.isdir("training_test_{}".format(filetag)):
                print(' weights exist for this training set, would you like to load them?')
                print(' enter "y" if yes')
                print(' enter "q" to quit')
                print(' press anything else to continue with fresh model')
                load = input()
                if load == 'y':
                    model.load_weights(checkpoint_path)
                    print('weights loaded')
                elif load == 'q':
                    print('exiting...')
                    quit()

            hist_file = '{}/history_{}'.format(checkpoint_dir,filetag)
            model_file = '{}/model_{}.h5'.format(checkpoint_dir,filetag)

            if validation:
                history = model.fit_generator(generator = training_generator, validation_data = validation_generator, epochs = epochs,callbacks=[cp_callback])
            else:
                history = model.fit_generator(generator = training_generator,epochs = epochs,callbacks=[cp_callback])
            try:
                results = scipy.io.loadmat(hist_file)
                print('loading training history')
                results['generator'] = np.hstack([results['generator'], history.history['loss']])
                print('history combined')
            except:
                results = dict([('generator',history.history['loss'])])

            scipy.io.savemat(hist_file,results)
            print('\nhistory saved\n')
            model.save(model_file)
            print('\nmodel saved\n')


# Append all of these files names to the database of models I have trained
dm.updateFileList(dm.filenames,label,files)



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
