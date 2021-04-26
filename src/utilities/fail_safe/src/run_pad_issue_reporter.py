#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
"""
When people click "Report Issue" button in the pad, pad will send an mqtt
message. This program receives this message and then post issue at JiRA.
"""
import argparse
import sys
from pad_issue_reporter import PadIssueReporter


def main():
    """Prog entry"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mqtt-fqdn", default="192.168.1.3")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    args = parser.parse_args()

    pad_issue_reporter = PadIssueReporter(args.mqtt_fqdn, args.mqtt_port)
    pad_issue_reporter.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
