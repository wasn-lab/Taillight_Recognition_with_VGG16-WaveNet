#!/usr/bin/env python
"""
Check if global variables are comform to coding style:
    1. under_score, and
    2. starts with g_
"""
import argparse
import sys
import logging
import os
import subprocess
import re
from ci_utils import (get_affected_files, get_compile_command,
                      is_external_package)
__DECL_RE = re.compile(
    r"\|-VarDecl 0x[0-9a-f]+.* <line:(?P<line_no>\d+).+ (used )?"
    r"(?P<var_name>[_a-zA-Z][_a-zA-Z0-9]*) "
    r"'(?P<decl_type>[^']+)'.*")
__DECL_RE_SAME_LINE = re.compile(
    r"\|-VarDecl 0x[0-9a-f]+.* (used )?"
    r"(?P<var_name>[_a-zA-Z][_a-zA-Z0-9]*) "
    r"'(?P<decl_type>[^']+)'.*")


def __get_global_var_decls(cpp):
    decls = []
    cmd = get_compile_command(cpp)
    if not cmd:
        return decls
    if "clang++" not in cmd[0]:
        logging.warning("clang++ is needed for analyzing %s", cpp)
        return 0
    for idx, item in enumerate(cmd):
        if "\\\"" in item:
            cmd[idx] = item.replace("\\", "")
    cmd += ["-Xclang", "-ast-dump", "-fno-color-diagnostics"]
    logging.debug(" ".join(cmd))
    try:
        output = subprocess.check_output(cmd).strip().decode("utf-8")
    except subprocess.CalledProcessError:
        output = ""
    for line in output.splitlines():
        if not line.startswith("|-VarDecl"):
            continue
        if "extern" in line:
            continue
        if " referenced " in line:
            continue
        decls.append(line)
    return decls


def _is_const(_type):
    if "const " in _type:
        return True
    return False


def _parse_var_decl(decl):
    ret = {"ast_repr": decl,
           "line": "undetected"}
    match = __DECL_RE.match(decl)
    if match:
        ret["line"] = match.expand(r"\g<line_no>")
    else:
        match = __DECL_RE_SAME_LINE.match(decl)

    if not match:
        logging.warning("Not match to regular expression: %s", decl)
        return {}
    ret["var"] = match.expand(r"\g<var_name>")
    ret["decl_type"] = match.expand(r"\g<decl_type>")
    ret["is_const"] = _is_const(ret["decl_type"])
    # ret["actual_type"] = match.expand("\g<actual_type>")
    return ret


def _is_global_var_naming(var_name):
    if not var_name.startswith("g_"):
        return False
    for idx, _ch in enumerate(var_name):
        if _ch.isupper() and var_name[idx-1].islower():
            return False

    return True


def check_cpp_global_var_naming(cpp):
    """
    Use clang-generated ast to find global variable naming violations.
    """
    if not os.path.isfile(cpp):
        return
    if not cpp.endswith(".cpp"):
        return
    if is_external_package(cpp):
        return
    logging.info("Check global variable naming convention: %s", cpp)
    violations = 0
    for var_decl in [_parse_var_decl(_) for _ in __get_global_var_decls(cpp)]:
        if var_decl.get("is_const", True):
            continue
        _var = var_decl.get("var", "")
        _type = var_decl.get("decl_type", "")
        _line = var_decl.get("line", "")
        if _is_global_var_naming(_var):
            logging.info("PASS: %s (line: %s, type: %s)", _var, _line, _type)
        else:
            violations += 1
            logging.warning(
                "FAIL: %s: global variable name is not under_score style "
                "with starting g_ (line: %s, type: %s, AST repr: %s)",
                _var, _line, _type, var_decl.get("ast_repr", ""))
    logging.info("Number of violations: %d", violations)
    return 0


def main():
    """Prog entry"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp", required=True)
    args = parser.parse_args()
    return check_cpp_global_var_naming(args.cpp)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
