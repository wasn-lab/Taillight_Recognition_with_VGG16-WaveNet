#!/bin/sh

class_1=no_signal
class_2=turn_right

file_1=no_signal2.mp4 # class 1 file
file_2=turn_right2.mp4 # class 2 file

seq_sec=1 # sequence in second

time_1=`ffmpeg -i ${file_1} 2>&1 | grep Duration | awk '{print $2}' | head -c 8`

hour_1=`echo ${time_1} | awk 'BEGIN {FS=":"} {print $1}'`
min_1=`echo ${time_1} | awk 'BEGIN {FS=":"} {print $2}'`
sec_1=`echo ${time_1} | awk 'BEGIN {FS=":"} {print $3}'`
sec_1_all=$((hour_1*60*60 + min_1*60 + sec_1))

time_2=`ffmpeg -i ${file_2} 2>&1 | grep Duration | awk '{print $2}' | head -c 8`
hour_2=`echo ${time_2} | awk 'BEGIN {FS=":"} {print $1}'`
min_2=`echo ${time_2} | awk 'BEGIN {FS=":"} {print $2}'`
sec_2=`echo ${time_2} | awk 'BEGIN {FS=":"} {print $3}'`
sec_2_all=$(( hour_2*60*60 + min_2*60 + sec_2))




seq_cnt_1=$((sec_1_all / seq_sec))
seq_cnt_2=$((sec_2_all / seq_sec))

echo seq_cnt_1=${seq_cnt_1}
echo seq_cnt_2=${seq_cnt_2}

exit;

for(( i=0 ; i<seq_cnt_1 ; i=i+1))
do
	start=$((i * seq_sec))
	echo "ffmpeg -i ${file_1} -ss 00:00:0${start} -t 00:00:02 -strict -2 data/train/${class_1}/${class_1}_train_1.mp4"
	echo "ffmpeg -i ${file_1} -ss 00:00:0${start} -t 00:00:02 -strict -2 data/test/${class_1}/${class_1}_test_1.mp4"
done

for(( i=0 ; i<seq_cnt_2 ; i=i+1))
do
	start=$((i * seq_sec))
	echo "ffmpeg -i ${file_2} -ss 00:00:0${start} -t 00:00:02 -strict -2 data/train/${class_2}/${class_2}_train_1.mp4"
	echo "ffmpeg -i ${file_2} -ss 00:00:0${start} -t 00:00:02 -strict -2 data/test/${class_2}/${class_2}_test_1.mp4"
done
