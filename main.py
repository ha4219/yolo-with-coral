import pickle
import numpy


with open('data/pred.npy', 'rb') as f:
    preds = pickle.load(f)
with open('data/label.npy', 'rb') as f:
    labels = pickle.load(f)


for pred in preds:
    print(pred)