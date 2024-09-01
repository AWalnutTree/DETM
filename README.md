# DETM - Adapted for use with Marketing Journals

Credit to Dieng et al:


This is code that accompanies the paper titled "The Dynamic Embedded Topic Model" by Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei. (Arxiv link: https://arxiv.org/abs/1907.05545).

The DETM is an extension of the Embedded Topic Model (https://arxiv.org/abs/1907.04907) to corpora with temporal dependencies. The DETM models each word with a categorical distribution whose parameter is given by the inner product between the word embedding and an embedding representation of its assigned topic at a particular time step. The word embeddings allow the DETM to generalize to rare words. The DETM learns smooth topic trajectories by defining a random walk prior over the embeddings of the topics. The DETM is fit using structured amortized variational inference with LSTMs.

Data scraping & processing is achieved seperately in: 
https://github.com/AWalnutTree/JM_Corpus







<!-- Current goals:

Implement split paragraphs true setting
Further corpus cleaning
 - articles only
 - find and fill any missing parts
 - remove even more un-important words
 - curate some stopwords that perhaps should not be stopwords
read up on literature to find a better set of parameters for further testing
create different min_df settings and test them
create documentation for documentation done for everything up to now 
continue literature study
 - About the model & best practices in configuring it
 - about proper ways of dealing with irregularities found while working with the model 


possible steps:

configure for slurm & cluster based run
configure for gpu based run -->


