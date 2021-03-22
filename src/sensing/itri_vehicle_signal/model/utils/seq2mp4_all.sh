


seq_dir_list=`find test/ -name "*light_mask*"| rev | cut -d'/' -f2- | rev`

#echo ${seq_dir_list}

for seq_dir in ${seq_dir_list}
do
	echo seq_dir=${seq_dir}
	seq_name=`echo ${seq_dir} | rev | awk 'BEGIN {FS="/"} {print $1}' | rev`
	echo seq_name=${seq_name}

	python3 img2mp4.py ${seq_dir} ./${seq_name}.mp4

done


# echo "find ${PATH}/ -name "*light_mask*""

# seq_dir_list=`find . -name "*light_mask*"`

# echo ${seq_dir_list}

# #SEQS=`find . -name "light_mask" | grep OOO | rev | cut -d'/' -f2- | rev`

# #python3 img2mp4.py test/aug_seq_03_test-02-26-2016_11-12-02_OLR_10028/ ./test.mp4


