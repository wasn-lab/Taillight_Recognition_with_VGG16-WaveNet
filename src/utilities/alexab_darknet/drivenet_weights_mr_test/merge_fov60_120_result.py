#!/usr/bin/env python
import argparse
import io
import logging
import json
import os
import sys

def _read_json_file(jfile):
    with io.open(jfile, encoding="utf-8") as _fp:
        contents = _fp.read()
    return json.loads(contents)


def _rewrite_image_urls(docs):
    for doc in docs:
        for field in ["expect", "actual"]:
            # Rewrite full path like to a shorter form:
            #   fov60/1596507098735017792_expect.jpg
            comps = doc[field].split("/")
            doc[field] = "/".join(comps[5:])


def _get_base_url(docs):
    base = "http://ci.itriadv.co/"
    if docs:
        filename = docs[0]["filename"]
        fields = filename.split("/")[2:5]
        base = base + "/".join(fields)
    return base


def _merge_check_result(artifacts_dir, commit_id, branch_name, repo_status):
    docs = []
    for angle in ["fov60", "fov120"]:
        jfile = os.path.join(artifacts_dir, angle, "check_weights_result.json")
        if not os.path.isfile(jfile):
            continue
        docs += _read_json_file(jfile)
    num_violations = sum(_["num_violations"] for _ in docs)
    _rewrite_image_urls(docs)
    result = {"test_cases": docs,
              "type": "DM.003",  # used by backend when posting to it.
              "num_violations": num_violations,
              "result": "PASS",
              "repo_status": repo_status,
              "commit_id": commit_id,
              "branch_name": branch_name,
              "job_url": _get_base_url(docs),
              }
    if num_violations > 0:
        result["result"] = "FAIL"
    output_file = os.path.join(artifacts_dir, "check_result.json")
    with io.open(output_file, "w", encoding="utf-8") as _fp:
        _fp.write(json.dumps(result, sort_keys=True))
    logging.warn("Write %s", output_file)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--commit-id", default="abcd123456")
    parser.add_argument("--branch-name", default="unspecified")
    parser.add_argument("--repo-status", default="dirty")
    args = parser.parse_args()
    result = _merge_check_result(args.artifacts_dir, args.commit_id, args.branch_name, args.repo_status)
    if result["num_violations"] > 0:
        logging.error("Find unexpected detection!")
    return result["num_violations"]


if __name__ == "__main__":
    sys.exit(main())
