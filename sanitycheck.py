import os

print("The text is visible if the clusters can run files in DETM")
print()
print("Now it Should print things from the embeddings folder")
with open("embeddings/un/skipgram_emb_300d.txt", 'rb') as f:
    x = 0
    for l in f:
        line = l.decode().split()
        word = line[0]
        if x < 20:
            print(word)
            x+=1
print()
