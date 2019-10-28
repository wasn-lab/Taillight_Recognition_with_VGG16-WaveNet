#!/bin/bash
set -x
readonly repo_dir=$(git rev-parse --show-toplevel)
cp ${repo_dir}/scripts/hooks/pre-receive ${repo_dir}/.git/hooks/pre-receive
cp ${repo_dir}/scripts/hooks/pre-commit ${repo_dir}/.git/hooks/pre-commit

