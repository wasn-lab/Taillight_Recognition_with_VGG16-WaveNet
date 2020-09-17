import argparse
from fail_safe_checker import FailSafeChecker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ini", default="fail_safe.ini")
    args = parser.parse_args()
    checker = FailSafeChecker(args.ini)
    checker.run()

if __name__ == "__main__":
    main()
