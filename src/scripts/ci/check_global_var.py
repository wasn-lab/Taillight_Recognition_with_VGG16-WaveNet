#!/usr/bin/env python
"""
Check if global variables are comform to coding style:
    1. under_score, and
    2. starts with g_
"""
import sys
import logging
import os
import subprocess
import re
from ci_utils import get_affected_files, get_compile_command, is_external_package
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
    logging.info(" ".join(cmd))
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
    print("Check global variable naming convention: " + cpp)
    for var_decl in [_parse_var_decl(_) for _ in __get_global_var_decls(cpp)]:
        _var = var_decl.get("var", "")
        _type = var_decl.get("decl_type", "")
        _line = var_decl.get("line", "")
        if _is_global_var_naming(_var):
            print("PASS: {} (line: {}, type: {})".format(_var, _line, _type))
        else:
            logging.warning(
                "FAIL: %s: global variable name is not under_score style "
                "with starting g_ (line: %s, type: %s, AST repr: %s)",
                _var, _line, _type, var_decl.get("ast_repr", ""))
    print("")

    return 0

def check_global_var_naming():
    """
    Return the number of naming violations
    """
    num_violations = 0
    for cpp in get_affected_files():
        if not os.path.isfile(cpp):
            continue
        if not cpp.endswith(".cpp"):
            continue
        if is_external_package(cpp):
            continue
        num_violations += check_cpp_global_var_naming(cpp)
    return num_violations


def main():
    """Prog entry"""
    return check_global_var_naming()


if __name__ == "__main__":
    sys.exit(main())
