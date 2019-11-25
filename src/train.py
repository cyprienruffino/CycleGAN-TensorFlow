import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import sys

from utils.config import loadconfig


def do_run(filepath, dataA_path, dataB_path, checkpoint_resume_path=None, epoch=0):
    name = filepath.split('/')[-1].replace('.py', '')
    config = loadconfig(filepath)

    print("\nRunning " + name + "\n")

    if not os.path.exists("runs"):
        os.mkdir("runs")

    checkpoints_dir = os.path.join("runs", name, "checkpoints")
    logs_dir = os.path.join("runs", name, "logs")

    if checkpoint_resume_path is None:
        os.mkdir(os.path.join("runs", name))
        os.mkdir(checkpoints_dir)
        shutil.copy2(filepath, os.path.join("runs", name, "config.py"))

    model = config.model(config, dataA_path, dataB_path, logs_dir, checkpoints_dir, checkpoint_resume_path, epoch=epoch)
    model.train()
    model.reset_session()


def main():
    if len(sys.argv) < 4:
        print("Usage : python train.py config_file dataA_path dataB_path [path_to_checkpoints] [epoch]")
        exit(1)

    filepath = sys.argv[1]
    dataA_path = sys.argv[2]
    dataB_path = sys.argv[3]

    if len(sys.argv) == 5:
        print("Usage : python train.py config_file dataA_path dataB_path [path_to_checkpoints] [epoch]")
        exit(1)

    checkpoint_resume_path = None
    epoch = 0

    if len(sys.argv) == 6:
        checkpoint_resume_path = sys.argv[4]
        epoch = int(sys.argv[5])

    if os.path.isfile(filepath):
        do_run(filepath, dataA_path, dataB_path, checkpoint_resume_path, epoch)
    else:
        for run in os.listdir(filepath):
            if ".py" in run:
                do_run(filepath + "/" + run, dataA_path, dataB_path)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
