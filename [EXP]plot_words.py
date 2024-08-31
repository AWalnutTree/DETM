#******************************************************************#
# plot_words.py
# This is to test plotting word embeddings using PCA.

# WORK IN PROGRESS
#******************************************************************#
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import numpy as np
import scipy.io
from sklearn.decomposition import PCA
import pickle

CLI = argparse.ArgumentParser()
CLI.add_argument(
    "--corpus",
    type=str,
    default="JMR",
    help="Corpus to use (JM or JMR)"
)
CLI.add_argument(
    "--dim",
    type=str,
    default="2",
    help="Dimension of PCA (2 or 3)"
)
CLI.add_argument(
    "--words",
    type=str,
    nargs='*',
    help="List of specific words to plot (space-separated)"
)

args = CLI.parse_args()

use = 'JM' if args.corpus == 'JM' else 'JMR'

if use == 'JM':
    beta = scipy.io.loadmat('./results/detm_jm_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_10_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_beta.mat')['values']
    timestamps = 'data/JM/split_paragraph_False/min_df_10/timestamps.pkl'
    data_file = 'data/JM/split_paragraph_False/min_df_10'
    emb_path = 'embeddings/JM/skipgram_emb_300d.txt'
    shift_value = 1936
else:
    beta = scipy.io.loadmat('./results/detm_jmr_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_10_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_beta.mat')['values']
    timestamps = 'data/JMR/split_paragraph_False/min_df_10/timestamps.pkl'
    data_file = 'data/JMR/split_paragraph_False/min_df_10'
    emb_path = 'embeddings/JMR/skipgram_emb_300d.txt'
    shift_value = 1963

def main():

    # Load vocabulary
    with open(f'{data_file}/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        
    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, 300))
    
    # Load embeddings
    with open(emb_path, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if word in vocab:
                vect = np.array(line[1:]).astype(float)
                embeddings[vocab.index(word)] = vect

    # Filter words if specified
    words_to_plot = args.words if args.words else vocab

    # Perform PCA
    pca = PCA(n_components=int(args.dim))
    pca_result = pca.fit_transform(embeddings)

    # Plotting
    if args.dim == "2":
        plt.figure(figsize=(10, 8))
        for i, word in enumerate(vocab):
            if word in words_to_plot:
                plt.scatter(pca_result[i, 0], pca_result[i, 1], marker='o', s=5, alpha=0.7)
                plt.annotate(word, (pca_result[i, 0], pca_result[i, 1]), fontsize=8, alpha=0.7)
                
        plt.title(f'PCA of Word Embeddings ({args.corpus}) - 2D')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.show()
    
    elif args.dim == "3":
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i, word in enumerate(vocab):
            if word in words_to_plot:
                ax.scatter(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], marker='o', s=5, alpha=0.7)
                ax.text(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], word, fontsize=8, alpha=0.7)
        
        ax.set_title(f'PCA of Word Embeddings ({args.corpus}) - 3D')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.show()

if __name__ == "__main__":
    main()
