#!/usr/bin/env python3
import progressbar
from tensorboard.backend.event_processing import event_accumulator as ea
import os
import pandas as pd
import sys


def tbtocsv(logpath, output_file):
    acc = ea.EventAccumulator(logpath)
    acc.Reload()

    tags = acc.Tags()["scalars"]
    data = {}

    for tag in tags:
        if 'MSE_train' not in tag and 'G_cost' not in tag and 'D_cost' not in tag:
            _, values = zip(*[(s.step, s.value) for s in acc.Scalars(tag)])
            data[tag] = values

    pd.DataFrame(data=data).to_csv(output_file)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        if os.path.isdir(sys.argv[1]):
            path = sys.argv[1]
            bar = progressbar.ProgressBar(maxvalue=len(os.listdir(path)), redirect_stdout=True)
            for runpath in bar(os.listdir(path)):
                for filepath in os.listdir(path + os.sep + runpath + os.sep + "logs" + os.sep):
                    if "events" in filepath:
                        tbtocsv(path + os.sep + runpath + os.sep + "logs" + os.sep + filepath, sys.argv[2] + runpath + ".csv")
        else:
            tbtocsv(sys.argv[1], sys.argv[2])
    else:
        print("Usage : tbtocsb.py log output")
