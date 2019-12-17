import scipy.io
import matplotlib.pyplot as plt
import pdb
import DataManagement as DM
import SKELETON_LAYOUT as SK
import numpy as np

# plot_set = 'batch_sizes'
# plot_set = 'layer_sizes'
# plot_set = 'joint_nums'
# plot_set = 'dec7_retrain'

#results = dict([('generator',history.history['loss'])])
fig,axes  = plt.subplots(1,1) #len(filenames[plot_set]),,sharey = True
fig.set_size_inches(15,7,forward = True)
for plot_set in ['dec8_validated','dec7_retrain']:
    for i,f in enumerate(DM.filenames[plot_set]):
        file = f + '/' + 'history_' + f[f.find('_s')+1::]
        file2 = f + '/' + 'history_' + f
        try:
            #everything = scipy.io.loadmat(file)
            history = scipy.io.loadmat(file)['generator'][0]
            #pdb.set_trace()
        except:
            history = scipy.io.loadmat(file2)['generator'][0]
        if plot_set == 'batch_sizes':
            axes.plot(history, label =f[f.find('0_')+2::])
        if plot_set == 'layer_sizes':
            axes.plot(history, label =f[f.find('4_')+2:f.find('512')-1])
        if plot_set == 'joint_nums' or plot_set == 'dec7_retrain' or plot_set == 'dec8_validated':
            subs,jnts,layer,batch = DM.filename2params(f)
            try:
                if (jnts == np.array(SK.WRIST_LEFT)).all():
                    set = 'Left Arm'
                elif (jnts == SK.ARM_LEFT).all():
                    set = 'Left wrist'
            except:
                set = 'full body (31)'
            axes.plot(history, label = '{}{}'.format(plot_set,layer) +' nodes: ' +set)

axes.grid()
axes.legend(loc = 'upper right')
plt.title('Loss curves for {}'.format(plot_set))
plt.xlabel('epochs')
plt.ylabel('L2 loss')
#pdb.set_trace()
plt.show()
