#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys


def plotscalars(csvspath, scalar, param_name, output_file):

    scalars = []
    params = []
    for csv in os.listdir(csvspath):
        df = pd.read_csv(csvspath + os.sep + csv)

        x = df[scalar].tolist()
        curparam = float(csv.split("_")[-1].replace(".csv", ""))

        scalars.append(np.min(x))
        params.append(curparam)

    z = list(zip(params, scalars))
    z.sort(key=lambda x: x[0])
    paramss, scalarss = zip(*z)

    plt.xlabel(param_name)
    plt.ylabel(scalar)
    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.plot(paramss, scalarss, "bo", paramss, scalarss, "b-")
    plt.savefig(output_file)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        plotscalars(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Usage : scalars.py logs_path scalar param_name output_file")
