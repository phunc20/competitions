"""
Let's remove the chord `N` to see if we can get the cheated score even higher
"""

import json
import pandas as pd
import os

D = dict()

train_dir = "/home/phunc20/datasets/aidea/2021-01-09-和絃辨識/CE200/"
for thing in os.listdir(train_dir):
    subdir = os.path.join(train_dir, thing)
    if os.path.isfile(subdir):
        continue
    gt_file = os.path.join(subdir, "ground_truth.txt")
    df = pd.read_csv(gt_file, header=None, delimiter="\t")
    # df.values is ndarray and ndarray is not serializable (thus cannot be an element of json)
    #D[thing] = df.values.tolist()
    D[thing] = df[df[2] != 'N'].values.tolist()

#json.dump(D, fp="cheat.json")
with open("cheat_no_N.json", 'w') as f:
    json.dump(D, fp=f)



