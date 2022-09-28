import os 
import glob
import shutil
from tqdm import tqdm


label = ["OOO", "OLO", "OOR", "OLR", "BOO", "BLO", "BOR", "BLR", "BBB"]

dir_path = '../data/rear_dataset/' 
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

pbar = tqdm(total=len(glob.glob('./rear_train/*/*light_mask/')))
for file in glob.glob('./rear_train/*/*light_mask/'):
    # print(file)
    for obj_label in label :
        if obj_label in file:
            # print(file)
            file_split_dir = file.split('\\')
            # print(os.path.join(dir_path, file_split_dir[2]+file_split_dir[1]))
            shutil.copytree(file, os.path.join(dir_path, obj_label, file_split_dir[1],file_split_dir[2]))
    pbar.set_description("Processing %s" % file_split_dir[1][-9:])
    pbar.update(1)
pbar.close()