#******************************************************************#
# plot_documents.py
# This script is used to plot the topic components for each document.
# WORK IN PROGRESS
#******************************************************************#
import scipy.io
import argparse
import pickle

CLI = argparse.ArgumentParser() 
CLI.add_argument(
    "--corpus",
    type=str,
    default="JMR",
    help="Corpus to use (JM or JMR)"
)

args = CLI.parse_args()
use = 'JM' if args.corpus == 'JM' else 'JMR'

if use == 'JM':
    beta = scipy.io.loadmat('./results/detm_jm_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_10_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_beta.mat')['values']
    alpha = scipy.io.loadmat('./results/detm_jm_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_10_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_alpha.mat')['values']
    timestamps = 'data/JM/split_paragraph_False/min_df_10/timestamps.pkl'
    data_file = 'data/JM/split_paragraph_False/min_df_10'
    shift_value = 1936
else:
    beta = scipy.io.loadmat('./results/detm_jmr_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_10_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_beta.mat')['values']
    alpha = scipy.io.loadmat('./results/detm_jmr_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_10_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_alpha.mat')['values']
    timestamps = 'data/JMR/split_paragraph_False/min_df_10/timestamps.pkl'
    data_file = 'data/JMR/split_paragraph_False/min_df_10'
    shift_value = 1963

# Print out the topic components for each document
print('beta: ', beta.shape)
print('alpha: ', alpha.shape)

# Load timestamps
with open(timestamps, 'rb') as f:
    timelist = pickle.load(f)

# Number of topics (K), timestamps (T), vocabulary size (V)
K, T, V = beta.shape
