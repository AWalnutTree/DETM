
import scipy.io
import data 


corp = 'JM'

if corp == 'JMR':
    beta = scipy.io.loadmat('./results/detm_jmr_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_10_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_beta.mat')['values']
    data_file = 'data/JMR/split_paragraph_False/min_df_10'
else:
    beta = scipy.io.loadmat('./results/detm_jm_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_10_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_beta.mat')['values']
    data_file = 'data/JM/split_paragraph_False/min_df_10'
# print('beta: ', beta.shape)

## get vocab
vocab, train, valid, test = data.get_data(data_file, temporal=True)
vocab_size = len(vocab)

num_words = 12
times = range(44)
num_topics = 50
topic_to_print = 27  # Specify the topic number to print

print(f"Printing topic {topic_to_print} of {corp}")

for t in times:
    gamma = beta[topic_to_print, t, :]
    top_words = list(gamma.argsort()[-num_words+1:][::-1])
    topic_words = [vocab[a] for a in top_words]
    print('Topic {} .. Time: {} ===> {}'.format(topic_to_print, t, topic_words))