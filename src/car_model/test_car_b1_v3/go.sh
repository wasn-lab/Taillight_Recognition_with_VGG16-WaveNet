set -x
for f in `ls`; do
  sed -i 's/B1_V2/B1_V3/g' $f
done
