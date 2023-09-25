import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("out", help="Path for the results to be saved to in CSV format.", type=str)
args = parser.parse_args()
output_path = args.out
if not output_path.endswith(".csv"):
    output_path += ".csv"

if os.path.isfile(output_path):
    print("Warning: output is overwriting an existing file [enter to continue]")
    input()

GOAL_SENTENCES_PATH = "goal_sentences.txt"
DATA_PATH = "data/"

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

if not os.path.isfile(GOAL_SENTENCES_PATH):
    print()
    print("ERROR: Goal sentences file not found!")
    print(f"Please provide a new-line separated list of sentences in {GOAL_SENTENCES_PATH}.")
    quit()

if not os.path.isdir(DATA_PATH):
    print()
    print("ERROR: Data folder not found!")
    print(f"Please put data text files in the folder {DATA_PATH}.")
    quit()

with open(GOAL_SENTENCES_PATH, "r") as f:
    goal_sentences = ".".join(f.readlines())
    goal_sentences = sentence_split(goal_sentences)
    print("Using the following goal sentences:")
    print("\n".join(goal_sentences))

print()
print("Encoding goal sentences...")
goal_encs = model.encode(goal_sentences)
# goal_enc = np.mean(goal_encs, axis=0)
# goal_enc /= np.linalg.norm(goal_enc)

all_sentences = []

for file in os.listdir(DATA_PATH):
    if not file.endswith(".txt"):
        print(f"Skipping '{file}' as not a .txt file...")
        continue

    print(f"Reading file '{file}'...")
    path = os.path.join(DATA_PATH, file)

    with open(path, "r") as f:
        text = "\n".join(f.readlines())
        sentences = sentence_split(text)
        print(len(sentences), "sentences")

        all_sentences += sentences

print()
print(f"Finished reading data, {len(all_sentences)} sentences total.")
print()
print("Encoding sentences...")
encs = model.encode(sentences)
print("Computing scores...")
scores = np.matmul(goal_encs, encs.T)
print()

# Mean score = mean similarity over all goal sentences
mean_scores = np.mean(scores, axis=0)

# Print all sentences in descending mean score order
pairs = list(zip(list(mean_scores), range(len(sentences))))
pairs = sorted(pairs, key=lambda a:-a[0])
print(f"Writing to file {output_path}...")
with open(output_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(("Score","Index","Sentence"))
    for score,ind in pairs:
        writer.writerow((str(score),str(ind),f"\"{sentences[ind]}\""))
        # print(score,"\t",sentences[ind])
print("Done")

# Display a graph showing the similarity score for each goal sentence
xs = list(range(len(sentences)))
for i in range(len(goal_sentences)):
    plt.plot(xs, scores[i], label=goal_sentences[i])
plt.legend()
plt.show()



