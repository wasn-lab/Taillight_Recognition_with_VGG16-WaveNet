#!/usr/bin/env python
"""
Check if commit files contain symbolic links.
"""
import sys
import logging
import os
from ci_utils import get_affected_files


def _check_symbolic_links(affected_files):
    num_links = 0
    for fname in affected_files:
        if not os.path.isfile(fname):
            continue
        if os.path.islink(fname):
            logging.error("Do not commit symbolic links: %s", fname)
            num_links += 1
    return num_links


def main():
    """Prog entry"""
    return _check_symbolic_links(get_affected_files())


if __name__ == "__main__":
    sys.exit(main())
