#-*- coding: utf-8 -*-
import numpy as np
import scipy.io.wavfile as wav
from sklearn.decomposition import FastICA

RATE =  44100

def decompose_normal(files):
    decompose_partial(files,len(files))

def decompose_partial(files,n):
    mix = [wav.read(files[i])[1] for i in range(len(files))]
    
    S = np.c_[mix[:n]] #先頭n個のみを用いる
    S = S.astype(np.float64)
    S /= S.std(axis = 0)

    ica = FastICA(n_components=n)
    S_ = ica.fit_transform(S)
    names = [('./output/sep' + str(i+1) + '_p') for i in range(n)]

    filenames = [(names[i] + '.wav') for i in range(n)]
    for i in range(n):
        wav.write(filenames[i],RATE,S_[:,i])

def decompose_withnoise(files,noise,weights):
    np.random.seed(0)
    mix = np.array([wav.read(files[i])[1] for i in range(len(files))])
    S = np.c_[mix[0][0:430000]]
    for i in range(1,len(files)):
        S = np.c_[S,mix[i][0:430000]]
    #S = np.c_[mix1[0:430000],mix2[0:430000],mix3[0:430000],mix4[0:430000]]
    S = S.astype(np.float64)
    S += noise*np.random.normal(size=S.shape)
    S /= S.std(axis = 0)
    
    S=np.dot(S,weights.T)
    
    names = [('mix' + str(i+1) + '_n') for i in range(len(weights))]
    
    filenames = [('./output/' + names[i] + '.wav') for i in range(len(weights))]
    for i in range(len(weights)):
        wav.write(filenames[i],RATE,S[:,i])
    
    ica = FastICA(n_components=len(weights))
    S_ = ica.fit_transform(S)
    #S_100 = S_ * 300 #音量調節
    
    names = [('./output/out' + str(i+1) + '_n') for i in range(len(weights))]
    
    filenames = [(names[i] + '.wav') for i in range(len(weights))]
    for i in range(len(weights)):
        wav.write(filenames[i],RATE,S_[:,i])
    '''
    filenames = [(names[i] + '_100.wav') for i in range(3)]
    for i in range(3):
        wav.write(filenames[i],RATE,S_100[:,i])
    '''

if __name__ == '__main__':
    files_y = ['mix5_1_y.wav', 'mix5_2_y.wav', 'mix5_3_y.wav', 'mix5_4_y.wav', 'mix5_5_y.wav']
    files_o = ['input1_o.wav', 'input2_o.wav', 'input3_o.wav', 'input4_o.wav'] 
    decompose_partial(files_y, 4)

    weights = np.array([[1.0,1.0,1.0,-0.5],[1.0,1.0,-1.0,0.5],[1.0,-1.0,1.0,0.5],[-1.0,1.0,1.0,0.5]])
    noise = 5000
    decompose_withnoise(files_o,noise,weights)

