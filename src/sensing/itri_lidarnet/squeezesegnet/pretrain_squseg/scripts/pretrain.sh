#!/bin/bash

if [ $# -eq 0 ]
then
  echo "Usage: ../pretrain_squseg/scripts/pretrain.sh [options]"
  echo " "
  echo "options:"
  echo "-h, --help                show brief help"
  echo "-dataset                  filename under /src/lidar_squseg_detect/data/"
  echo "-view_type                view type for phi partition"
  exit 0
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ../pretrain_squseg/scripts/pretrain.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-dataset                  filename under /src/lidar_squseg_detect/data/"
      echo "-view_type                view type for phi partition"
      exit 0
      ;;
    -dataset)
      DATASET="$2"
      shift
      shift
      ;;
    -view_type)
      VIEW_TYPE="$2"
      shift
      shift
      ;;
    *)
      break
      ;;
  esac
done


cd ../../lidar_squseg_detect/data/$DATASET/labelFile
rename 's/\.pcd$/\.txt/' *.pcd

cd  $OLDPWD
cd ../../lidar_squseg_detect/data/$DATASET
mkdir txtFile
chmod -R 777 txtFile

cd  $OLDPWD
cd ../../../../devel/lib/pretrain_squseg
./pretrain_squseg $DATASET $VIEW_TYPE

cd  $OLDPWD
# pip install tqdm
# pip install statistics
python convert.py --inpath ../../lidar_squseg_detect/data/$DATASET/txtFile/ --outpath ../../lidar_squseg_detect/data/$DATASET/ --outdir npyFile --conv npy --viewtype $VIEW_TYPE
cd ../../lidar_squseg_detect/data/$DATASET
chmod -R 777 npyFile
mkdir split_dataset

cd  $OLDPWD
cd ../../lidar_squseg_detect/data/$DATASET/npyFile
ls *P0.npy > ../split_dataset/all.txt
ls *P0.npy > ../split_dataset/train.txt
ls *P0.npy > ../split_dataset/val.txt

cd  $OLDPWD
