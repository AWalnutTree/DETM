import scipy.io 
import matplotlib.pyplot as plt 
import data 
import pickle 
import numpy as np 
import sys

def top_words(gamma, num_words):
    option = 2
    if option == 1:
        # top words at the last time point
        top_word_indices = gamma[-1, :].argsort()[-num_words:][::-1]  
    elif option == 2:
        # Average beta values across all timestamps
        avg_beta = gamma.mean(axis=0)  
        top_word_indices = avg_beta.argsort()[-num_words:][::-1]
    elif option == 3:
        # Max value word at each timestamp
        top_word_indices = set() 
    
        for t in range(gamma.shape[0]):
            top_indices_at_t = gamma[t, :].argsort()[-2:][::-1]
            top_word_indices.update(top_indices_at_t)
        
        top_word_indices = list(top_word_indices)
        
        if len(top_word_indices) > num_words:
            top_word_indices = top_word_indices[:num_words]
    elif option == 4:
        # Experimental Diverse Set
        top_word_indices = set()
        for t in range(gamma.shape[0]):
            top_word_indices.update(gamma[t, :].argsort()[-num_words:][::-1])
        return list(top_word_indices)[:num_words]
    elif option == 5:
        # Most Changed Words (Beginning vs End)
        change = np.abs(gamma[-1, :] - gamma[0, :])
        top_word_indices = change.argsort()[-num_words:][::-1]
    elif option == 6:
        # Most Changed Words (Cumulative Timestamp)
        cumulative_change = np.sum(np.abs(np.diff(gamma, axis=0)), axis=0)  # Shape: (V,)
        top_word_indices = cumulative_change.argsort()[-num_words:][::-1]



    return top_word_indices


arg = sys.argv[1]

use = 'JM' if arg == 'JM' else 'JMR'

# Load the beta matrix
if use == 'JM':
    beta = scipy.io.loadmat('./results/detm_jm_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_10_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_beta.mat')['values']
    timestamps = 'data/JM/split_paragraph_False/min_df_10/timestamps.pkl'
    data_file = 'data/JM/split_paragraph_False/min_df_10'
    shift_value = 1936
else:
    beta = scipy.io.loadmat('./results/detm_jmr_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_10_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_beta.mat')['values']
    timestamps = 'data/JMR/split_paragraph_False/min_df_10/timestamps.pkl'
    data_file = 'data/JMR/split_paragraph_False/min_df_10'
    shift_value = 1963
print('beta: ', beta.shape)

# Load the timelist
with open(timestamps, 'rb') as f:
    timelist = pickle.load(f)
print('timelist: ', timelist)
T = len(timelist)
ticks = [str(x) for x in timelist]
print('ticks: ', ticks)

ticks = [int(x) + shift_value for x in timelist]
print('ticks with shift: ', ticks)

# Get the vocabulary and other data
vocab, train, valid, test = data.get_data(data_file, temporal=True)
vocab_size = len(vocab)

# Plot settings
num_topics = 30
num_words = 10

# Increase the figure size and use fewer topics per row
fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(30, 15), dpi=70, facecolor='w', edgecolor='k')
axes = axes.flatten()

#Title for the plot
fig.suptitle(f'Top {num_words} words evolution for each topic in {use}', fontsize=16)

# Plot the top words' beta values evolution for each topic
for k in range(num_topics):
    gamma = beta[k, :, :]  # T x V

    ax = axes[k]
    colors = plt.cm.get_cmap('tab10', num_words)  # Use a colormap for better visualization

    #top_word_indices = gamma[-1, :].argsort()[-num_words:][::-1]  # Get indices of top words at the last time point
    top_word_indices = top_words(gamma, num_words)

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
#plt.savefig(f'word_evolution_{use}_all.png')
plt.show()