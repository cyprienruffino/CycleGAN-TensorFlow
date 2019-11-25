import os
import sys


def gen_exp(base_config, field, value_list, output_dir):
    with open(base_config, "r") as f:
        original = f.read()

    to_replace = None
    for item in original.split("\n"):
        if "self."+field in item:
            to_replace = item.strip()

    if to_replace is None:
        print(" Field not found")
        exit(0)

    field, _ = to_replace.split("=")
    field = field.strip()

    basename = base_config.split(os.path.sep)[-1][:-3]

    for value in value_list:
        edited = original.replace(to_replace, field + " = " + str(value))
        with open(output_dir + os.path.sep + basename + "_" + field.replace("self.", "") + "_" + value + ".py", "w") as f:
            f.write(edited)



if __name__ == "__main__":
    if len(sys.argv) == 5:
        gen_exp(sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4])
    else:
        print("Usage : gen_expe base_config field value_list output_dir")
