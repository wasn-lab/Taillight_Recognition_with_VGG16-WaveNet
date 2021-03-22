#!/bin/sh

rm data_file.csv test/ train/ -rf

python3 traverse.py train no_signal ./dataset/OOO
python3 traverse.py train turn_right ./dataset/OOR
python3 traverse.py train turn_left ./dataset/OLO
python3 traverse.py train flashers ./dataset/OLR
python3 traverse.py test no_signal ./dataset/OOO
python3 traverse.py test turn_right ./dataset/OOR
python3 traverse.py test turn_left ./dataset/OLO
python3 traverse.py test flashers ./dataset/OLR

