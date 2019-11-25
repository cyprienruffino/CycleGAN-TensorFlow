import os
import random
import shutil
import sys


def main(input_path, num_files, output_path):
    files = os.listdir(input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    random.shuffle(files)

    for i in range(num_files):
        shutil.copy2(os.path.join(input_path, files[i]), os.path.join(output_path, files[i]))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python copy_expe.py input_path num_files output_files")
    else:
        main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
