import re
import os
import io
import pprint
import pandas as pd
REPO_DIR = "/home/chtseng/repo/itriadv"
RGX = re.compile(r"\s*<name>(?P<pkg_name>[_\w]+)</name>")


def get_file_issues():
    df = pd.read_csv("report-by-files.csv")
    return [(REPO_DIR + df.at[i, "File"], df.at[i, "New"])
            for i in range(len(df))]


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


def find_pkgs():
    res = {}
    for root, __dirs, files in os.walk(REPO_DIR):
        for filename in files:
            if filename != "package.xml":
                continue
            pkg = _get_package_name(os.path.join(root, filename))
            if not pkg:
                continue
            res[pkg] = root
    return res


def locate_file_to_pkg(pkgs, filename):
    for pkg in pkgs:
        if filename.startswith(pkgs[pkg]):
            return pkgs[pkg]
    return None


def group_issues_by_pkg(pkgs, file_issues):
    report = {pkgs[_]: 0 for _ in pkgs}
    for fname, num_issues in file_issues:
        pkg = locate_file_to_pkg(pkgs, fname)
        report[pkg] += num_issues
    return report


def main():
    file_issues = get_file_issues()
    pkg_paths = find_pkgs()
    report = group_issues_by_pkg(pkg_paths, file_issues)
    pprint.pprint(report)
    total_issues = sum(report.values())
    print("Total issues: {}".format(total_issues))


if __name__ == "__main__":
    main()
