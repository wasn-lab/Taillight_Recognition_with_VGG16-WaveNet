from fail_safe_checker import FailSafeChecker


def main():
    checker = FailSafeChecker("nodes_info.ini")
    checker.run()

if __name__ == "__main__":
    main()
