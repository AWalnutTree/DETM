import scipy.io 
import matplotlib.pyplot as plt 
import data 
import pickle 
import numpy as np 

TOPIC = 35
TIME = 40
WORDSTOSHOW = 10  
SOURCE = 'data/un/split_paragraph_0/min_df_30'
BETA = 'results/SLURMED/detm_un_K_50_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_200_RhoSize_300_L_3_minDF_30_trainEmbeddings_1_beta.mat'

beta = scipy.io.loadmat(BETA)['values'] ## K x T x V

with open(f'{SOURCE}/timestamps.pkl', 'rb') as f:
    timelist = pickle.load(f)

T = len(timelist)
ticks = [str(x) for x in timelist]

## get vocab
data_file = SOURCE
vocab, train, valid, test = data.get_data(data_file, temporal=True)
vocab_size = len(vocab)  

gamma = beta[TOPIC, TIME, :]
top_words = list(gamma.argsort()[-WORDSTOSHOW+1:][::-1])
topic_words = [vocab[a] for a in top_words]
print(topic_words)
