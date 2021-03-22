OOO_SEQS=`find . -name "light_mask" | grep OOO | rev | cut -d'/' -f2- | rev`
OOR_SEQS=`find . -name "light_mask" | grep OOR | rev | cut -d'/' -f2- | rev`
OLO_SEQS=`find . -name "light_mask" | grep OLO | rev | cut -d'/' -f2- | rev`
OLR_SEQS=`find . -name "light_mask" | grep OLR | rev | cut -d'/' -f2- | rev`

OOO_TAR=$1
OOR_TAR=$2
OLO_TAR=$3
OLR_TAR=$4

for dir in ${OOO_SEQS}
do
	cp ${dir} ${OOO_TAR}/ -rf
done

for dir in ${OOR_SEQS}
do
	cp ${dir} ${OOR_TAR}/ -rf
done

for dir in ${OLO_SEQS}
do
	cp ${dir} ${OLO_TAR}/ -rf
done

for dir in ${OLR_SEQS}
do
	cp ${dir} ${OLR_TAR}/ -rf
done
