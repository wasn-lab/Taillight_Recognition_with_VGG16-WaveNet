#!/bin/sh


OOO_SEQS=`find . -name "light_mask" | grep OOO | rev | cut -d'/' -f2- | rev`
OOR_SEQS=`find . -name "light_mask" | grep OOR | rev | cut -d'/' -f2- | rev`
OLO_SEQS=`find . -name "light_mask" | grep OLO | rev | cut -d'/' -f2- | rev`
OLR_SEQS=`find . -name "light_mask" | grep OLR | rev | cut -d'/' -f2- | rev`
BOO_SEQS=`find . -name "light_mask" | grep BOO | rev | cut -d'/' -f2- | rev`
BOR_SEQS=`find . -name "light_mask" | grep BOR | rev | cut -d'/' -f2- | rev`
BLO_SEQS=`find . -name "light_mask" | grep BLO | rev | cut -d'/' -f2- | rev`
BLR_SEQS=`find . -name "light_mask" | grep BLR | rev | cut -d'/' -f2- | rev`

# for seq_dir in ${OOO_SEQS}
# do
	# python3 seq_augmentation.py ${seq_dir} 5
# done

for seq_dir in ${OOR_SEQS}
do
	python3 seq_augmentation.py ${seq_dir} 2
done

for seq_dir in ${OLO_SEQS}
do
	python3 seq_augmentation.py ${seq_dir} 1
done

for seq_dir in ${OLR_SEQS}
do
	python3 seq_augmentation.py ${seq_dir} 9
done

# for seq_dir in ${BOO_SEQS}
# do
	# python3 seq_augmentation.py ${seq_dir} 5
# done

for seq_dir in ${BOR_SEQS}
do
	python3 seq_augmentation.py ${seq_dir} 4
done

for seq_dir in ${BLO_SEQS}
do
	python3 seq_augmentation.py ${seq_dir} 2
done

for seq_dir in ${BLR_SEQS}
do
	python3 seq_augmentation.py ${seq_dir} 9
done
