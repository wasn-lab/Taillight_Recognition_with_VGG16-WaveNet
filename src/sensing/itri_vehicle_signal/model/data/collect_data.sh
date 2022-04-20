
python traverse.py train no_signal ./dataset/OOO
python traverse.py train turn_right ./dataset/OOR
python traverse.py train turn_left ./dataset/OLO
python traverse.py train flashers ./dataset/OLR
python traverse.py test no_signal ./dataset/OOO
python traverse.py test turn_right ./dataset/OOR
python traverse.py test turn_left ./dataset/OLO
python traverse.py test flashers ./dataset/OLR

python traverse.py train no_signal ./dataset/BOO
python traverse.py train turn_right ./dataset/BOR
python traverse.py train turn_left ./dataset/BLO
python traverse.py train flashers ./dataset/BLR
python traverse.py test no_signal ./dataset/BOO
python traverse.py test turn_right ./dataset/BOR
python traverse.py test turn_left ./dataset/BLO
python traverse.py test flashers ./dataset/BLR

echo finished 
read
