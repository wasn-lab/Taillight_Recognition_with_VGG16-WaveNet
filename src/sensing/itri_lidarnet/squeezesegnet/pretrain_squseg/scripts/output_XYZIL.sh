#!/bin/bash

if [ $# -eq 0 ]
then
  echo "Usage: ../pretrain_squseg/scripts/output_XYZIL.sh [options]"
  echo " "
  echo "options:"
  echo "-h, --help                show brief help"
  echo "-source_dir               given source directory xxx just above ori and OK"
  exit 0
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ../pretrain_squseg/scripts/output_XYZIL.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-source_dir               given source directory xxx just above ori and OK"
      exit 0
      ;;
    -source_dir)
      SRC_DIR="$2"
      shift
      shift
      ;;
    *)
      break
      ;;
  esac
done


SCRIPT_PATH=$(readlink -f "$0")
EXE_DIR=$(dirname "$SCRIPT_PATH")
echo $EXE_DIR

cd $SRC_DIR/OK

for i in $(ls); do 
if [ -d "$i" ];
then
cd $i; rename 's/\.pcd$/\.txt/' *.pcd; cd ..; 
fi
done

cd  ..
mkdir XYZIL

cd  $EXE_DIR
cd ../../../../devel/lib/pretrain_squseg
./output_XYZIL $SRC_DIR

cd  $EXE_DIR
