#!/usr/bin/env python
"""
List all package names in the repository.
"""
from __future__ import print_function
import pprint
import os
import io
import re
REPO_SRC = os.path.abspath(os.path.join(__file__, "..", "..", "..", ".."))
RGX = re.compile(r"\s*<name>(?P<pkg_name>[_\w]+)</name>")

def _get_xmls():
    ret = []
    for root, _, files in os.walk(REPO_SRC):
        for fname in files:
            if fname != "package.xml":
                continue
            ret.append(os.path.join(root, fname))
    return ret

def _get_package_name(xml_file):
    ret = ""
    with io.open(xml_file) as _fp:
        lines = _fp.read().splitlines()
    for line in lines:
        match = RGX.match(line)
        if match:
            ret = match.expand(r"\g<pkg_name>")
    if not ret:
        print("Cannot find package name: {}".format(xml_file))
    return ret

def main():
    """Prog entry."""
    xmls = _get_xmls()
    pkgs = {_get_package_name(xml):xml for xml in xmls}
    pprint.pprint(pkgs)
    for pkg in sorted(pkgs.keys()):
        print("{}".format(pkg))

if __name__ == "__main__":
    main()
