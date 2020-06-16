#!/usr/bin/env python3
import argparse
import pexpect
import time


def gen_drivenet_engine(launch_type):
    start_time = time.time()
    cmd = ["roslaunch", "drivenet", launch_type]
    print(" ".join(cmd))
    child = pexpect.spawnu(" ".join(cmd))

    child.expect("Loading Complete!", timeout=300)
    print(child.before)
    print(child.after)
    child.sendcontrol('c')  # kill child
    child.wait()
    elapsed_time = time.time() - start_time
    print("Takes {}s to generate engine".format(elapsed_time))


def main():
    """Prog entry"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-type", default="b1_v2_drivenet_group_a.launch")
    args = parser.parse_args()
    gen_drivenet_engine(args.launch_type)

if __name__ == "__main__":
    main()
