#******************************************************************#
# plot_all_jmr.py
# This script is used to plot the evolution of all topics and their 
# top words in the DETM model for the JMR dataset. 

# USAGE:
# ||$ python plot_all_jmr.py
#******************************************************************#
import scipy.io 
import matplotlib.pyplot as plt 
import data 
import pickle 
import numpy as np 

BS = 10

# Load the beta matrix
beta = scipy.io.loadmat(f'./results/newstuff/detm_JMR_K_50_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_200_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_beta.mat')['values']
#beta = scipy.io.loadmat('./results/detm_jmr_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_10_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_beta.mat')['values']
#beta = scipy.io.loadmat(f'./results/S-batchcheck/detm_jmr_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_{BS}_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_beta.mat')['values']
print('beta: ', beta.shape)

# Load the timelist
with open('data/JMRnew/split_paragraph_False/min_df_10/timestamps.pkl', 'rb') as f:
    timelist = pickle.load(f)
print('timelist: ', timelist)
T = len(timelist)
ticks = [str(x) for x in timelist]
print('ticks: ', ticks)

shift_value = 1963  # Change this to whatever shift you want
ticks = [int(x) + shift_value for x in timelist]
print('ticks with shift: ', ticks)

# Get the vocabulary and other data
data_file = 'data/JMRnew/split_paragraph_False/min_df_10'
vocab, train, valid, test = data.get_data(data_file, temporal=True)
vocab_size = len(vocab)

# Plot settings
num_topics = 30
num_words = 10

# Increase the figure size and use fewer topics per row
fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(30, 15), dpi=70, facecolor='w', edgecolor='k')
axes = axes.flatten()

# Plot the top words' beta values evolution for each topic
for k in range(num_topics):
    gamma = beta[k, :, :]  # T x V

    ax = axes[k]
    colors = plt.cm.get_cmap('tab10', num_words)  # Use a colormap for better visualization

    top_word_indices = gamma[-1, :].argsort()[-num_words:][::-1]  # Get indices of top words at the last time point

    for i, word_idx in enumerate(top_word_indices):
        ax.plot(range(T), gamma[:, word_idx], color=colors(i), lw=2, linestyle='--', marker='o', markersize=4, label=vocab[word_idx])
    
    ax.set_title('Topic {}'.format(k), fontsize=12)
    ax.set_xticks(range(0, T, max(1, T // 10)))  # Fewer ticks
    ax.set_xticklabels(ticks[::max(1, T // 10)], rotation=45)
    ax.set_xlabel('Time')
    ax.set_ylabel('Beta Value')

    # Use a smaller legend to prevent overlapping
    ax.legend(fontsize='xx-small', loc='upper left', frameon=False, bbox_to_anchor=(1, 1))

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(f'word_evolution_jmr_{BS}.png')
plt.show()