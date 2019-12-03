#!/bin/bash

if [ $# -eq 0 ]
then
  echo "Usage: ../pretrain_squseg/scripts/pretrain.sh [options]"
  echo " "
  echo "options:"
  echo "-h, --help                show brief help"
  echo "-source_dir               given source directory xxx just above ori and OK"
  echo "-view_type                view type for phi partition"
  echo "-file_head                add header to output filename to avoid differnt pcd with the same filename (ROS time)"
  exit 0
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ../pretrain_squseg/scripts/pretrain.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-source_dir               given source directory xxx just above ori and OK"
      echo "-view_type                view type for phi partition"
      echo "-file_head                add header to output filename to avoid differnt pcd with the same filename (ROS time)"
      exit 0
      ;;
    -source_dir)
      SRC_DIR="$2"
      shift
      shift
      ;;
    -view_type)
      VIEW_TYPE="$2"
      shift
      shift
      ;;
    -file_head)
      FILE_HEAD="$2"
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
mkdir txtFile

cd  $EXE_DIR
cd ../../../../devel/lib/pretrain_squseg
./pretrain_squseg_v2 $SRC_DIR $VIEW_TYPE $FILE_HEAD

cd  $EXE_DIR
# pip install tqdm
# pip install statistics
python convert.py --inpath $SRC_DIR/txtFile/ --outpath $SRC_DIR/ --outdir npyFile --conv npy --viewtype $VIEW_TYPE
cd $SRC_DIR
mkdir split_dataset

cd $SRC_DIR/npyFile
ls *P0.npy > ../split_dataset/all.txt
ls *P0.npy > ../split_dataset/train.txt
ls *P0.npy > ../split_dataset/val.txt

cd  $EXE_DIR
python MeanStdEstimate.py --inpath $SRC_DIR/npyFile/ --viewtype $VIEW_TYPE
