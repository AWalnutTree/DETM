#******************************************************************#
# plot_word_evo_un.py
# This script was used for visualization on the UN dataset.
# DEPRECATED
#******************************************************************#
import scipy.io 
import matplotlib.pyplot as plt 
import data 
import pickle 
import numpy as np 

beta = scipy.io.loadmat('./results/SLURMED/detm_un_K_50_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_200_RhoSize_300_L_3_minDF_30_trainEmbeddings_1_beta.mat')['values'] ## K x T x V  #MODIFICATION
print('beta: ', beta.shape)

with open('data/un2/split_paragraph_1/min_df_30/timestamps.pkl', 'rb') as f: #MODIFICATION 'un' -> 'data/un/split_paragraph_1'
    timelist = pickle.load(f)
print('timelist: ', timelist)
T = len(timelist)
ticks = [str(x) for x in timelist]
print('ticks: ', ticks)

## get vocab
data_file = 'data/un2/split_paragraph_1/min_df_30' #MODIFICATION 'un' -> 'data/un/split_paragraph_1'
vocab, train, valid, test = data.get_data(data_file, temporal=True)
vocab_size = len(vocab)

## plot topics 
num_words = 5 #10 -> 5
times = [0, 30, 40] #40 -> 30
num_topics = 50
for k in range(num_topics):
    for t in times:
        gamma = beta[k, t, :]
        top_words = list(gamma.argsort()[-num_words+1:][::-1])
        #print(f"length of vocab:{0}, length of top_words:{1}".format(len(vocab), len(top_words)))
        topic_words = [vocab[a] for a in top_words]
        print('Topic {} .. Time: {} ===> {}'.format(k, t, topic_words)) 

print('Topic Climate Change...')
num_words = 10
for t in range(46):
    gamma = beta[46, t, :]
    top_words = list(gamma.argsort()[-num_words+1:][::-1])
    topic_words = [vocab[a] for a in top_words]
    print('Time: {} ===> {}'.format(t, topic_words)) 

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9), dpi=80, facecolor='w', edgecolor='k')
ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flatten()
ticks = [str(x) for x in timelist]
#plt.xticks(np.arange(T)[0::10], timelist[0::10])

words_1 = ['security', 'palestinian', 'dignity', 'decades', 'led', 'population', 'neighbouring', 'sovereignty']
tokens_1 = [vocab.index(w) for w in words_1]
betas_1 = [beta[34, :, x] for x in tokens_1]
for i, comp in enumerate(betas_1):
    ax1.plot(range(T), comp, label=words_1[i], lw=2, linestyle='--', marker='o', markersize=4)
ax1.legend(frameon=False)
# print('np.arange(T)[0::10]: ', np.arange(T)[0::10])
ax1.set_xticks(np.arange(T)[0::10])
ax1.set_xticklabels(timelist[0::10])
ax1.set_title('Topic 34', fontsize=12)

words_2 = ['organizations', 'international', 'world', 'country', 'rational', 'trade']
tokens_2 = [vocab.index(w) for w in words_2]
betas_2 = [beta[41, :, x] for x in tokens_2]
for i, comp in enumerate(betas_2):
    ax2.plot(range(T), comp, label=words_2[i], lw=2, linestyle='--', marker='o', markersize=4)
ax2.legend(frameon=False)
ax2.set_xticks(np.arange(T)[0::10])
ax2.set_xticklabels(timelist[0::10])
ax2.set_title('Topic 41', fontsize=12)

words_3 = ['weapons', 'multilateral', 'developing', 'situations', 'red', 'successful', 'withdrawal']
tokens_3 = [vocab.index(w) for w in words_3]
betas_3 = [beta[47, :, x] for x in tokens_3]
for i, comp in enumerate(betas_3):
    ax3.plot(range(T), comp, label=words_3[i], lw=2, linestyle='--', marker='o', markersize=4)
ax3.legend(frameon=False)
ax3.set_xticks(np.arange(T)[0::10])
ax3.set_xticklabels(timelist[0::10])
ax3.set_title('Topic 47', fontsize=12)

words_4 = ['terrorism', 'promoting', 'plan', 'regular', 'summit', 'line', 'somalia']
tokens_4 = [vocab.index(w) for w in words_4]
betas_4 = [beta[49, :, x] for x in tokens_4]
for i, comp in enumerate(betas_4):
    ax4.plot(range(T), comp, label=words_4[i], lw=2, linestyle='--', marker='o', markersize=4)
ax4.legend(frameon=False)
ax4.set_xticks(np.arange(T)[0::10])
ax4.set_xticklabels(timelist[0::10])
ax4.set_title('Topic 49', fontsize=12)

words_5 = ['mankind', 'bold', 'accordance', 'contribution', 'middle', 'dialogue']
tokens_5 = [vocab.index(w) for w in words_5]
betas_5 = [beta[34, :, x] for x in tokens_5]
for i, comp in enumerate(betas_5):
    ax5.plot(range(T), comp, label=words_5[i], lw=2, linestyle='--', marker='o', markersize=4)
ax5.legend(frameon=False)
ax5.set_xticks(np.arange(T)[0::10])
ax5.set_xticklabels(timelist[0::10])
ax5.set_title('Topic 34', fontsize=12)

plt.show()


