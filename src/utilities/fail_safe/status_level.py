#!/usr/bin/env python
"""
Definition for Fail-Safe levels:
- OK: Everything works fine.
- WARN: Ego does not have sufficient confidence in perception/decision
- ERROR: Some modules go wrong, but ego can drive for a few seconds.
- FATAL: Something serious happens. Need to stop ego car immedidately.

The level code increments by 10 for reserving the possible future extension.
"""

OK = 0
WARN = 10
ERROR = 20
FATAL = 30
UNKNOWN = -1
