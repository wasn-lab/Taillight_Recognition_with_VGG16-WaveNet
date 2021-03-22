#!/bin/sh

if [  $# -ne 4 ] 
then
	echo "Usage:"
	echo "	classify_all.sh  [sequence length]  [class limit]  [saved model]   [video file path]"
	echo "	Example:"
	echo "		clasify_all.sh 30 4 data/checkpoints/lstm-features.030-0.007.hdf5 ./test_video/"
fi

seq_length=$1
class_limit=$2
saved_model=$3
video_file_path=$4

video_list=`find ${video_file_path} -name "*.mp4"`

echo cmd="find ${video_file_path} -name *.mp4"

echo "aaa~"

echo "video_list=${video_list}"


for video in ${video_list}
do
	log_file=${video}_log.txt
	error_file=${video}_error.txt

	echo "classify video ==>> $video ..."

	echo "    sequence length: ${seq_length}"
	echo "    class limit: ${class_limit}"
	echo "    saved model: ${saved_model}"
	echo "    video: ${video}"

	#echo "    cmd: python3 clasify.py ${seq_length} ${class_limit} ${saved_model} ${video}"
	#echo "log_file=${log_file}"
	#echo "error_file=${error_file}"

	start=`date +%s.%N`

	python3 clasify.py ${seq_length} ${class_limit} ${saved_model} ${video} 2>${error_file} 1> ${log_file}
	end=`date +%s.%N`

	cat ${log_file}

	runtime=$( echo "$end - $start" | bc -l )
	echo "    runtime: ${runtime}"

	printf "\n"
done



