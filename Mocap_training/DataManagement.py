import numpy as np
import pdb
import scipy.io
import pickle
# original list at bottom can be uncommeneted to load filenames


def filename2params(filename):
    substr = filename[filename.find('_s')+2:filename.find('_j')]
    subjects = str2array(substr)
    jntstr = filename[filename.find('_j')+2:filename.find('_l')]
    joints = str2array(jntstr)
    layer_size = int(filename[filename.find('_l')+2:filename.find('_b')])
    batch_size = int(filename[filename.find('_b')+2:filename.find('_f')])
    try:
        fps = int(filename[filename.find('_f')+2::])
    except:
        fps = 60
    return subjects,joints,layer_size,batch_size, fps

def params2filename(tag,subjects,joints,layer_size,batch_size, fps = 60):
    substr = array2str(subjects,'s')
    jntstr = array2str(joints,'j')
    laystr = '_l{}'.format(layer_size)
    btchstr = '_b{}'.format(batch_size)
    fpsstr = '_f{}'.format(fps)
    filename = tag + substr + jntstr + laystr + btchstr + fpsstr
    return filename

def str2array(str):
    s = np.array([], dtype = 'int')
    for sub in str.split('_'):
        nums = sub.split('-')
        if len(nums) == 1:
            s = np.hstack([s,[int(nums[0])]])
        else:
            s = np.hstack( [s, range( int(nums[0]), int(nums[1]) +1 )] )
    return s

def array2str(array,tag):
    array = np.sort(array)
    str = '_{}'.format(tag)
    addon = '{}'.format(array[0])
    Base = 0
    i = 0
    while True:
        #print(i,array[i])
        if i >= len(array)-1:
            if not i-Base == 0:
                addon += '-{}'.format(array[i])
            str += addon;
            break
            #pdb.set_trace()
        elif array[i+1] == array[i] + 1:
            i += 1
            continue
        if not i-Base == 0:
            addon += '-{}'.format(array[i])
        str += addon; i+= 1
        Base = i; addon ='_{}'.format(array[Base])
    return str

def updateFileList(filelist,key,filenames):
    if key in filelist:
        filelist[key] = np.append(filenames)
    else:
        filelist[key] = filenames
    with open('filelist.pickle', 'wb') as handle:
        pickle.dump(filenames,handle)
    return


#used once to gather metadata now probably not needed
def countTrials(last_subject = 143):
    trialCount = 0
    subject = 1
    trial = 1
    numTrials = []
    fpsTrials = []
    numSamples = []
    last = 0
    while subject<= last_subject:
        #print('here')
        trial = 1
        samplecount = []
        thisFPS = []
        while True:
            #print('inner')
            ret,temp,fps = loadMocapFromMAT(PATH,subject,trial)
            if ret:
                print('s: {}  t: {}  c: {}'.format(subject,trial,trialCount+1))
                trialCount += 1
                samplecount.append(len(temp))
                thisFPS.append(fps)
                trial += 1
            else:
                numTrials.append(trialCount-last)
                numSamples.append(samplecount)
                fpsTrials.append(thisFPS)
                last = trialCount
                subject += 1
                break
    return numTrials,fpsTrials,numSamples
# filenames = dict ([
#     ('batch_sizes',
#         ['batch_test_1-24_35-48_50_128',
#         'batch_test_1-24_35-48_50_256',
#         'batch_test_1-24_35-48_50_512',
#         'batch_test_1-24_35-48_50_1024',
#         'batch_test_1-24_35-48_50_2048']  ),
#
#     ('layer_sizes',
#         ['layer_test_1-24_40_512',
#         'layer_test_1-24_100_512',
#         'layer_test_1-24_480_512',
#         'layer_test_1-24_1440_512',
#         'layer_test_1-24_4320_512']  ),
#
#     ('joint_nums',
#         ['joint_tests_s1-24_j20_l40_b512',
#         'joint_tests_s1-24_j17-21_l40_b512',
#         'joint_tests_s1-24_j20_l100_b512',
#         'joint_tests_s1-24_j17-21_l100_b512',
#         'layer_test_s1-24_j1-31_l100_b512'   ]  ),
#
#     ('dec7_retrain',
#         ['dec7_retrain_s1-24_j20_l40_b512',
#         'dec7_retrain_s1-24_j17-21_l40_b512',
#         'dec7_retrain_s1-24_j20_l100_b512',
#         'dec7_retrain_s1-24_j17-21_l100_b512',
#         'layer_test_s1-24_j1-31_l100_b512'   ]  ),
#     ('dec8_validated',
#         ['dec8_validated_s1-24_j20_l40_b512',
#         'dec8_validated_s1-24_j17-21_l40_b512',
#         'dec8_validated_s1-24_j20_l100_b512',
#         'dec8_validated_s1-24_j17-21_l100_b512',
#         'layer_test_s1-24_j1-31_l100_b512'   ]  )
#     ])
#
# with open('filelist.pickle', 'wb') as handle:
#     pickle.dump(filenames,handle)


with open('filelist.pickle', 'rb') as handle:
    filenames = pickle.load(handle)
