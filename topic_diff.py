import scipy.io 
import matplotlib.pyplot as plt 
import data 
import pickle 
import numpy as np 

TOPIC = 14
TIME = 26
TOPWORDS = 30
SOURCE = 'data/JMR/split_paragraph_False/min_df_10'
BETA = 'results/S-batchcheck/detm_jmr_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_100_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_beta.mat'

beta = scipy.io.loadmat(BETA)['values'] ## K x T x V

with open(f'{SOURCE}/timestamps.pkl', 'rb') as f:
    timelist = pickle.load(f)

T = len(timelist)
ticks = [str(x) for x in timelist]

## get vocab
data_file = SOURCE
vocab, train, valid, test = data.get_data(data_file, temporal=True)
vocab_size = len(vocab)  

gamma = beta[TOPIC, :, :]

BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

retained_words_count_list = []
retained_words_ratio_list = []
new_words_count_list = []
new_words_ratio_list = []
removed_words_count_list = []
removed_words_ratio_list = []


for i in range(T):
    top_words = list(gamma[i, :].argsort()[-TOPWORDS+1:][::-1])
    topic_words = [vocab[a] for a in top_words]

    if i != 0:
        prev_top_words_id = list(gamma[i-1, :].argsort()[-TOPWORDS+1:][::-1])
        prev_top_words = [vocab[a] for a in prev_top_words_id]

        new_words = [word for word in topic_words if word not in prev_top_words]
        retained_words = [word for word in topic_words if word in prev_top_words]
        removed_words = [word for word in prev_top_words if word not in topic_words]

        new_words = [f"{GREEN}{word}{RESET}" for word in new_words]
        retained_words = [f"{BOLD}{word}{RESET}" for word in retained_words]
        removed_words = [f"{RED}{word}{RESET}" for word in removed_words]

        print(f'Time: {i-1} ==> {i}')

        retained_words_ratio = f'{100*len(retained_words)/TOPWORDS:.2f}%'
        retained_words_ratio_list.append(retained_words_ratio)
        retained_words_count = len(retained_words)
        retained_words_count_list.append(retained_words_count)
        retained_words_str = f'{BOLD}Retained words: {", ".join(retained_words)} {BOLD} ~{retained_words_count}, ~{retained_words_ratio} {RESET}'
        print(retained_words_str)

        new_words_ratio = f'{100*len(new_words)/TOPWORDS:.2f}%'
        new_words_ratio_list.append(new_words_ratio)
        new_words_count = len(new_words)
        new_words_count_list.append(new_words_count)
        new_words_str = f'{GREEN}New words: {", ".join(new_words)} {GREEN} +{new_words_count}, +{new_words_ratio} {RESET}'
        print(new_words_str)

        removed_words_ratio = f'{100*len(removed_words)/TOPWORDS:.2f}%'
        removed_words_ratio_list.append(removed_words_ratio)
        removed_words_count = len(removed_words)
        removed_words_count_list.append(removed_words_count)
        removed_words_str = f'{RED}Removed words: {", ".join(removed_words)} {RED} -{removed_words_count}, -{removed_words_ratio} {RESET}'
        print(removed_words_str)

print()
print('Average Retained Words Count: ', np.mean(retained_words_count_list))
print('Average New Words Count: ', np.mean(new_words_count_list))   
print('Average Removed Words Count: ', np.mean(removed_words_count_list))