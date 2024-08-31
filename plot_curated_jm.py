#******************************************************************#
# plot_curated_jm.py    
# This script visualizes the evolution of selected topics and words
# in the DETM model for the JM dataset to construct a insightful
# visualization of the model's behavior.

# USAGE: 
# ||$ python plot_curated_jm.py
#******************************************************************#
import scipy.io 
import matplotlib.pyplot as plt 
import data 
import pickle 
import numpy as np 

beta = scipy.io.loadmat('./results/detm_jm_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_10_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_beta.mat')['values'] ## K x T x V  #MODIFICATION
print('beta: ', beta.shape)

with open('data/JM/split_paragraph_False/min_df_10/timestamps.pkl', 'rb') as f: #MODIFICATION 'un' -> 'data/un/split_paragraph_1'
    timelist = pickle.load(f)
print('timelist: ', timelist)
T = len(timelist)
ticks = [str(x) for x in timelist]
print('ticks: ', ticks)

## get vocab
data_file = 'data/JM/split_paragraph_False/min_df_10' #MODIFICATION 'un' -> 'data/un/split_paragraph_1'
vocab, train, valid, test = data.get_data(data_file, temporal=True)
vocab_size = len(vocab)

## plot topics 
num_words = 20 #10 -> 5
times = [44] #40 -> 30
num_topics = 20
for k in range(num_topics):
    for t in times:
        gamma = beta[k, t, :]
        top_words = list(gamma.argsort()[-num_words+1:][::-1])
        print(f"length of vocab:{0}, length of top_words:{1}".format(len(vocab), len(top_words)))
        topic_words = [vocab[a] for a in top_words]
        print('Topic {} .. Time: {} ===> {}'.format(k, t, topic_words)) 

num_words = 10
# for t in range(46):
#     gamma = beta[46, t, :]
#     top_words = list(gamma.argsort()[-num_words+1:][::-1])
#     topic_words = [vocab[a] for a in top_words]
#     print('Time: {} ===> {}'.format(t, topic_words)) 

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9), dpi=80, facecolor='w', edgecolor='k')
ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flatten()
ticks = [str(x) for x in timelist]
#plt.xticks(np.arange(T)[0::10], timelist[0::10])

words_1 = ['patient', 'drug', 'health', 'insurance', 'medical']
tokens_1 = [vocab.index(w) for w in words_1]
betas_1 = [beta[1, :, x] for x in tokens_1]
for i, comp in enumerate(betas_1):
    ax1.plot(range(T), comp, label=words_1[i], lw=2, linestyle='--', marker='o', markersize=4)
ax1.legend(frameon=False)
print('np.arange(T)[0::10]: ', np.arange(T)[0::10])
ax1.set_xticks(np.arange(T)[0::10])
ax1.set_xticklabels(timelist[0::10])
ax1.set_title('Topic "Health related things"', fontsize=12)


words_5 = ['return', 'innovation', 'investor', 'stock', 'risk', 'asset']
tokens_5 = [vocab.index(w) for w in words_5]
betas_5 = [beta[3, :, x] for x in tokens_5]
for i, comp in enumerate(betas_5):
    ax2.plot(comp, label=words_5[i], lw=2, linestyle='--', marker='o', markersize=4)
ax2.legend(frameon=False)
ax2.set_xticks(np.arange(T)[0::10])
ax2.set_xticklabels(timelist[0::10])
ax2.set_title('Topic "Stonks"', fontsize=12)


words_11 = ['participant', 'hospital', 'health', 'psychological', 'target', 'simulation']
tokens_11 = [vocab.index(w) for w in words_11]
betas_11 = [beta[11, :, x] for x in tokens_11]
for i, comp in enumerate(betas_11):
    ax3.plot(comp, label=words_11[i], lw=2, linestyle='--', marker='o', markersize=4)
ax3.legend(frameon=False)
ax3.set_xticks(np.arange(T)[0::10])
ax3.set_xticklabels(timelist[0::10])
ax3.set_title('Topic "Health Research?"', fontsize=12)


words_13 = ['ad', 'channel', 'rating', 'online']
tokens_13 = [vocab.index(w) for w in words_13]
betas_13 = [beta[23, :, x] for x in tokens_13]
for i, comp in enumerate(betas_13):
    ax4.plot(comp, label=words_13[i], lw=2, linestyle='--', marker='o', markersize=4)
ax4.legend(frameon=False)
ax4.set_xticks(np.arange(T)[0::10])
ax4.set_xticklabels(timelist[0::10])
ax4.set_title('Topic "Online advertising?"', fontsize=12)

plt.savefig('word_evolution_jm_test.png')
plt.show()