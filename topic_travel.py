#******************************************************************#
# topic_travel.py
# This script is currently used to find the topic that moved the
# most and the least in the DETM model for the JM and JMR datasets.

# USAGE:
# ||$ python topic_travel.py --corpus [corpus] --method [method]

# Where [corpus] is the dataset to use (JM or JMR)
# Where [method] is the distance metric to use (cosine or euclidean)  
#******************************************************************#
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import scipy.io
import matplotlib.pyplot as plt
import argparse

CLI = argparse.ArgumentParser()
CLI.add_argument(
    "--corpus",
    type=str,
    default="JMR",
    help="Corpus to use (JM or JMR)"
)
CLI.add_argument(  
    "--method",
    type=str,
    default="cosine",
    help="Method to use (cosine or euclidean)"
)

args = CLI.parse_args()
corp = args.corpus
method = args.method


if corp == 'JMR':
    beta = scipy.io.loadmat('./results/detm_jmr_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_10_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_beta.mat')['values']
else:
    beta = scipy.io.loadmat('./results/detm_jm_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.001_Bsz_10_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_beta.mat')['values']
# print('beta: ', beta.shape)

# Assuming `beta` is already loaded as a K x T x V matrix (Topics x Time x Vocabulary)
K, T, V = beta.shape

# Initialize an array to hold the total travel distance for each topic
topic_distances = np.zeros(K)

# Calculate the travel distance for each topic
for k in range(K):
    topic_distance = 0.0
    for t in range(1, T):
        if method == 'euclidean':
            distance = euclidean(beta[k, t, :], beta[k, t-1, :])
        else:
            distance = cosine(beta[k, t, :], beta[k, t-1, :])
        topic_distance += distance
    topic_distances[k] = topic_distance

# Identify topics that moved the most and least
most_travel_topic = np.argmax(topic_distances)
least_travel_topic = np.argmin(topic_distances)

print (f"Results for {corp} using {method} distance:")
print(f"Topic {most_travel_topic} moved the most with a distance of {topic_distances[most_travel_topic]:.2f}.")
print(f"Topic {least_travel_topic} moved the least with a distance of {topic_distances[least_travel_topic]:.2f}.")

# plt.figure(figsize=(12, 6))
# plt.bar(range(K), topic_distances)
# plt.xlabel('Topic')
# plt.ylabel('Travel Distance')
# plt.title(f'Travel Distance for Each Topic in {corp}')
# plt.show()
