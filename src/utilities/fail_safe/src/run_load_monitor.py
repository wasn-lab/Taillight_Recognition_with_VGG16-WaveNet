# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import sys
from load_monitor import LoadMonitor


def main():
    monitor = LoadMonitor()
    monitor.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
