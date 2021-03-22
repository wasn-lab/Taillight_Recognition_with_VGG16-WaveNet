#!/bin/bash

TEST_SEQ_COUNT=10

mkdir -p train test



OOO_SEQS=`find . -name "light_mask" | grep OOO | rev | cut -d'/' -f2- | rev`
OOR_SEQS=`find . -name "light_mask" | grep OOR | rev | cut -d'/' -f2- | rev`
OLO_SEQS=`find . -name "light_mask" | grep OLO | rev | cut -d'/' -f2- | rev`
OLR_SEQS=`find . -name "light_mask" | grep OLR | rev | cut -d'/' -f2- | rev`
BOO_SEQS=`find . -name "light_mask" | grep BOO | rev | cut -d'/' -f2- | rev`
BOR_SEQS=`find . -name "light_mask" | grep BOR | rev | cut -d'/' -f2- | rev`
BLO_SEQS=`find . -name "light_mask" | grep BLO | rev | cut -d'/' -f2- | rev`
BLR_SEQS=`find . -name "light_mask" | grep BLR | rev | cut -d'/' -f2- | rev`


OOO_COUNT=`echo ${OOO_SEQS} | wc -w`
OOR_COUNT=`echo ${OOR_SEQS} | wc -w`
OLO_COUNT=`echo ${OLO_SEQS} | wc -w`
OLR_COUNT=`echo ${OLR_SEQS} | wc -w`
BOO_COUNT=`echo ${BOO_SEQS} | wc -w`
BOR_COUNT=`echo ${BOR_SEQS} | wc -w`
BLO_COUNT=`echo ${BLO_SEQS} | wc -w`
BLR_COUNT=`echo ${BLR_SEQS} | wc -w`

echo OOO_COUNT=${OOO_COUNT}
echo OOR_COUNT=${OOR_COUNT}
echo OLO_COUNT=${OLO_COUNT}
echo OLR_COUNT=${OLR_COUNT}
echo BOO_COUNT=${BOO_COUNT}
echo BOR_COUNT=${BOR_COUNT}
echo BLO_COUNT=${BLO_COUNT}
echo BLR_COUNT=${BLR_COUNT}


for(( i=0 ; i<${TEST_SEQ_COUNT} ; i=i+1))
do
	rand=${RANDOM}
	rand=$((rand+1))
	rand_set=${rand_set}" ${RANDOM}"
done


seq=${OOO_SEQS}
seq_cnt=${OOO_COUNT}
new_rand_set=""
for rand in ${rand_set}
do
	new_rand=$((rand % seq_cnt)); new_rand_set="${new_rand_set} ${new_rand} ";
done
echo new_rand_set=${new_rand_set}
for(( i=1 ; i<=${seq_cnt} ; i=i+1))
do
	#echo i=${i}

	seq_dir=`echo ${seq} | awk -v var=${i} '{print $var}'`
	#echo seq_dir=${seq_dir}

	rand=$(( rand % seq_cnt + 1))

	if [[ ${new_rand_set} == *" ${i} "* ]]; then
		target_seq_dir=test
	else
		target_seq_dir=train
	fi

	cp -rf `echo ${seq} | awk -v var=${i} '{print $var}'` ${target_seq_dir}
done


seq=${OOR_SEQS}
seq_cnt=${OOR_COUNT}
new_rand_set=""
for rand in ${rand_set}
do
	new_rand=$((rand % seq_cnt)); new_rand_set="${new_rand_set} ${new_rand} ";
done
echo new_rand_set=${new_rand_set}
for(( i=1 ; i<=${seq_cnt} ; i=i+1))
do
	#echo i=${i}

	seq_dir=`echo ${seq} | awk -v var=${i} '{print $var}'`
	#echo seq_dir=${seq_dir}

	rand=$(( rand % seq_cnt + 1))

	if [[ ${new_rand_set} == *" ${i} "* ]]; then
		target_seq_dir=test
	else
		target_seq_dir=train
	fi

	cp -rf `echo ${seq} | awk -v var=${i} '{print $var}'` ${target_seq_dir}
done


seq=${OLO_SEQS}
seq_cnt=${OLO_COUNT}
new_rand_set=""
for rand in ${rand_set}
do
	new_rand=$((rand % seq_cnt)); new_rand_set="${new_rand_set} ${new_rand} ";
done
echo new_rand_set=${new_rand_set}
for(( i=1 ; i<=${seq_cnt} ; i=i+1))
do
	#echo i=${i}

	seq_dir=`echo ${seq} | awk -v var=${i} '{print $var}'`
	#echo seq_dir=${seq_dir}

	rand=$(( rand % seq_cnt + 1))

	if [[ ${new_rand_set} == *" ${i} "* ]]; then
		target_seq_dir=test
	else
		target_seq_dir=train
	fi

	cp -rf `echo ${seq} | awk -v var=${i} '{print $var}'` ${target_seq_dir}
done



seq=${OLR_SEQS}
seq_cnt=${OLR_COUNT}
new_rand_set=""
for rand in ${rand_set}
do
	new_rand=$((rand % seq_cnt)); new_rand_set="${new_rand_set} ${new_rand} ";
done
echo new_rand_set=${new_rand_set}
for(( i=1 ; i<=${seq_cnt} ; i=i+1))
do
	#echo i=${i}

	seq_dir=`echo ${seq} | awk -v var=${i} '{print $var}'`
	#echo seq_dir=${seq_dir}

	rand=$(( rand % seq_cnt + 1))

	if [[ ${new_rand_set} == *" ${i} "* ]]; then
		target_seq_dir=test
	else
		target_seq_dir=train
	fi

	cp -rf `echo ${seq} | awk -v var=${i} '{print $var}'` ${target_seq_dir}
done

