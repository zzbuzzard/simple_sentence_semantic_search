import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Splits a piece of text into a list of sentences
def sentence_split(text):
    def flt(x):
        return len(x) > 5
    text = text.replace("\n", " ")  # New lines are replaced by spaces
    text = text.replace("  ", " ")  # Double spaces removed
    xs = text.split(".")
    xs = filter(flt, xs)            # Remove short sentences as they're probably erroneous
    return list(xs)

print("Loading model...") 

# See https://www.sbert.net/docs/pretrained_models.html for other available models
version = 'all-mpnet-base-v2'
model = SentenceTransformer(version)
model = model.eval()
print("Loaded!")

goal_sentences_path = input("Path to search sentences file: ")
data_path = input("Path to data file: ")

with open(goal_sentences_path, "r") as f:
    goal_sentences = ".".join(f.readlines())
    goal_sentences = sentence_split(goal_sentences)
    print("Using the following goal sentences:")
    print("\n".join(goal_sentences))

print()
print("Encoding goal sentences...")
goal_encs = model.encode(goal_sentences)
# goal_enc = np.mean(goal_encs, axis=0)
# goal_enc /= np.linalg.norm(goal_enc)

with open(data_path, "r") as f:
    text = "\n".join(f.readlines())
    sentences = sentence_split(text)
    print(len(sentences), "data sentences")
    print("Encoding...")
    encs = model.encode(sentences)
    print("Computing scores...")
    scores = np.matmul(goal_encs, encs.T)
    print()

    # Mean score = mean similarity over all goal sentences
    mean_scores = np.mean(scores, axis=0)

    # Print all sentences in descending mean score order
    pairs = list(zip(list(mean_scores), range(len(sentences))))
    pairs = sorted(pairs, key=lambda a:-a[0])
    for score,ind in pairs:
        print(score,"\t",sentences[ind])

    # Display a graph showing the similarity score for each goal sentence
    xs = list(range(len(sentences)))
    for i in range(len(goal_sentences)):
        plt.plot(xs, scores[i], label=goal_sentences[i])
    plt.legend()
    plt.show()



