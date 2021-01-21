# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
#!/usr/bin/env python
"""
Definition for Fail-Safe levels:
- OK: Everything works fine.
- WARN: Ego does not have sufficient confidence in perception/decision
- ERROR: Some modules go wrong, but ego can drive for a few seconds.
- FATAL: Something serious happens. Need to stop ego car immedidately.

The level code increments by 10 for reserving the possible future extension.
"""


# Used by fail-safe checker
OK = 0
WARN = 10
ERROR = 20
FATAL = 30
UNKNOWN = -1

STATUS_CODE_TO_STR = {
    OK: "OK",
    WARN: "WARN",
    ERROR: "ERROR",
    FATAL: "FATAL",
    UNKNOWN: "UNKNOWN"}

# Used by Wistron data exchange protocol, published in /vehicle/report/*
OFF = 0
FAULT = 1
ALARM = 2
NORMAL = 3
