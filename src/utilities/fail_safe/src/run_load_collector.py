#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
"""
Aggregate cpu/gpu loads and publish it
"""
from load_collector import LoadCollector


if __name__ == "__main__":
    col = LoadCollector()
    col.run()
