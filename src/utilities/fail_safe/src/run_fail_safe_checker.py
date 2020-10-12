import argparse
from fail_safe_checker import FailSafeChecker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid", default="b1", help="vehicle id")
    parser.add_argument("--ini", default="fail_safe.ini")
    parser.add_argument("--mqtt-ini", default="mqtt_b1_v2.ini")
    parser.add_argument("--debug-mode", action="store_true")
    args = parser.parse_args()
    checker = FailSafeChecker(args.vid, args.ini, args.mqtt_ini)
    checker.set_debug_mode(args.debug_mode)

    checker.run()

if __name__ == "__main__":
    main()
