#!/bin/sh


ooo_test_logs=`find ./test_video -name "*_log.txt" | grep OOO`
oor_test_logs=`find ./test_video -name "*_log.txt" | grep OOR`
olo_test_logs=`find ./test_video -name "*_log.txt" | grep OLO`
olr_test_logs=`find ./test_video -name "*_log.txt" | grep OLR`


for log in ${ooo_test_logs}
do
	result=success

	base_name=`basename ${log}`

	name=`echo ${base_name} | awk 'BEGIN {FS="."} {print $1}'`

	no_sig_avg=`cat ${log} | grep no_signal | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	turn_right_avg=`cat ${log} | grep turn_right | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	turn_left_avg=`cat ${log} | grep turn_left | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	flashers_avg=`cat ${log} | grep flashers | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	printf "${name} ${no_sig_avg} ${turn_right_avg} ${turn_left_avg} ${flashers_avg}\n"
done

printf "\n"

for log in ${oor_test_logs}
do
	result=success

	base_name=`basename ${log}`

	name=`echo ${base_name} | awk 'BEGIN {FS="."} {print $1}'`

	no_sig_avg=`cat ${log} | grep no_signal | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	turn_right_avg=`cat ${log} | grep turn_right | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	turn_left_avg=`cat ${log} | grep turn_left | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	flashers_avg=`cat ${log} | grep flashers | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	printf "${name} ${no_sig_avg} ${turn_right_avg} ${turn_left_avg} ${flashers_avg}\n"
done

printf "\n"

for log in ${olo_test_logs}
do
	result=success

	base_name=`basename ${log}`

	name=`echo ${base_name} | awk 'BEGIN {FS="."} {print $1}'`

	no_sig_avg=`cat ${log} | grep no_signal | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	turn_right_avg=`cat ${log} | grep turn_right | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	turn_left_avg=`cat ${log} | grep turn_left | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	flashers_avg=`cat ${log} | grep flashers | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	printf "${name} ${no_sig_avg} ${turn_right_avg} ${turn_left_avg} ${flashers_avg}\n"
done

printf "\n"

for log in ${olr_test_logs}
do
	result=success

	base_name=`basename ${log}`

	name=`echo ${base_name} | awk 'BEGIN {FS="."} {print $1}'`

	no_sig_avg=`cat ${log} | grep no_signal | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	turn_right_avg=`cat ${log} | grep turn_right | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	turn_left_avg=`cat ${log} | grep turn_left | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	flashers_avg=`cat ${log} | grep flashers | awk '{print $2}' | awk '{s+=$1; avg=s/NR} END {print avg}'`

	printf "${name} ${no_sig_avg} ${turn_right_avg} ${turn_left_avg} ${flashers_avg}\n"
done