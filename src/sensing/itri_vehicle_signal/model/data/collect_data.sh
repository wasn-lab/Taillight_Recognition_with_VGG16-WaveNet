
python traverse.py train no_signal ./label_dataset/OOO
python traverse.py train turn_right ./label_dataset/OOR
python traverse.py train turn_left ./label_dataset/OLO
python traverse.py train flashers ./label_dataset/OLR
python traverse.py test no_signal ./label_dataset/OOO
python traverse.py test turn_right ./label_dataset/OOR
python traverse.py test turn_left ./label_dataset/OLO
python traverse.py test flashers ./label_dataset/OLR

python traverse.py train no_signal ./label_dataset/BOO
python traverse.py train turn_right ./label_dataset/BOR
python traverse.py train turn_left ./label_dataset/BLO
python traverse.py train flashers ./label_dataset/BLR
python traverse.py test no_signal ./label_dataset/BOO
python traverse.py test turn_right ./label_dataset/BOR
python traverse.py test turn_left ./label_dataset/BLO
python traverse.py test flashers ./label_dataset/BLR

echo finished 
read
