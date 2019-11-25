import os
import sys
import shutil


def dup_exp(config_dir, n_times):
    for filename in os.listdir(config_dir):
        if ".py" in filename:
            for i in range(int(n_times)):
                shutil.copy2(config_dir + os.sep + filename, config_dir + os.sep + str(i) + "_" + filename)
            os.remove(config_dir + os.sep + filename)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        dup_exp(sys.argv[1], sys.argv[2])
    else:
        print("Usage : dup_expe configs_dir n_times")
