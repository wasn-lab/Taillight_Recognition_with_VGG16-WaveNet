import os 
import sys
import shutil

result_path = "./sort_dataset"

label = ["OOO", "OLO", "OOR", "OLR", "BOO", "BLO", "BOR", "BLR", "BBB"]


for root, dirs, files in os.walk("./taillight_image_dataset/result_data"):
    dir_paths = dirs
    break
# print(dir_paths)

for d_path in dir_paths:
    print(os.path.join("./taillight_image_dataset/result_data", d_path)+' start ...')
    folder = os.listdir(os.path.join("./taillight_image_dataset/result_data", d_path))
    for obj_p in folder:
        target_folder = os.path.join("./taillight_image_dataset/result_data", d_path, obj_p, 'light_mask')
        files = os.listdir(target_folder)
        # print(files)
        for obj_label in label:
            obj_list = [i for i in files if obj_label in i]
            if not obj_list:
                continue
            result_folder_name = d_path+'_'+obj_p+'_'+obj_label
            if not os.path.exists(os.path.join(result_path, result_folder_name, 'light_mask')):
                os.makedirs(os.path.join(result_path, result_folder_name, 'light_mask'))
            for o in obj_list :
                shutil.copy(os.path.join("./taillight_image_dataset/result_data", d_path, obj_p, 'light_mask', o), os.path.join(result_path, result_folder_name, 'light_mask'))
    print('Finished')
    
        # OOO_obj = [i for i in files if i[-7:-4] == "OOO"]
        # OLO_obj = [i for i in files if i[-7:-4] == "OLO"]
        # OOR_obj = [i for i in files if i[-7:-4] == "OOR"]
        # OLR_obj = [i for i in files if i[-7:-4] == "OLR"]
        # BOO_obj = [i for i in files if i[-7:-4] == "BOO"]
        # BLO_obj = [i for i in files if i[-7:-4] == "BLO"]
        # BOR_obj = [i for i in files if i[-7:-4] == "BOR"]
        # BLR_obj = [i for i in files if i[-7:-4] == "BLR"]
        # BBB_obj = [i for i in files if i[-7:-4] == "BBB"]
        # result_folder_name = d_path+'_'+obj_p+"_OOO"
        # if not os.path.exists(os.path.join(result_path, result_folder_name, 'light_mask')):
        #     os.makedirs(os.path.join(result_path, result_folder_name, 'light_mask'))
        # for o in OOO_obj :
        #     shutil.copy(os.path.join("./taillight_image_dataset/result_data", d_path, obj_p, 'light_mask', o), os.path.join(result_path, result_folder_name, 'light_mask'))