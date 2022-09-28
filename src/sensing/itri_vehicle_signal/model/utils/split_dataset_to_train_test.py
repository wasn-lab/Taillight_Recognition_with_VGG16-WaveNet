import os 
import glob
import sys
import random
import shutil
import numpy as np
from collections import defaultdict

import seq_augmentation

import argparse


parser = argparse.ArgumentParser()

# function
parser.add_argument('--aug', type=bool, default= False)
args = parser.parse_args()


dataset_path = "./rear_train"

label = ["OOO", "OLO", "OOR", "OLR", "BOO", "BLO", "BOR", "BLR", "BBB"]


# Counting every label need to augment 
def counting_label():
    print("Counting data amount")

    global dataset_path
    # dataset_path = "./test"
    global label

    num_label_list = {}

    for root, dirs, files in os.walk(dataset_path):
        for obj_label in label:
            obj_list = [i for i in dirs if obj_label in i]
            if not obj_list:
                continue
            print(obj_label+" has "+str(len(obj_list)))
            num_label_list[obj_label] = len(obj_list)
            # print(num_label_list)
            # print(max(num_label_list, key=num_label_list.get))

    aug_count = {}

    O_obj_num = { i : num_label_list[i] for i in label if i[0]=='O' and i in num_label_list }
    B_obj_num = { i : num_label_list[i] for i in label if i[0]=='B' and i in num_label_list }

    all_values = O_obj_num.values()
    max_value = max(all_values)
    # print(max_value)

    for key, value in O_obj_num.items():
        O_aug_count = int(max_value/value)
        aug_count[key] = O_aug_count
        # print(key+' need to aug : '+ str(O_aug_count))

    all_values = B_obj_num.values()
    max_value = max(all_values)

    for key, value in B_obj_num.items():
        B_aug_count = int(max_value/value)
        aug_count[key] = B_aug_count
        # print(key+' need to aug : '+ str(B_aug_count))

    # Remove the base augment number or it also will do augment
    
    for key, value in aug_count.items() :
        print(key+"  "+str(value))
        if value <=1 :
            aug_count[key] = 0

    return aug_count

# print(aug_count)


# Do seq_augmentation
def do_seq_augmentation(aug_count, aug_flag=False):
    if not aug_flag:
        return 

    global dataset_path
    # dataset_path = "./test"
    global label

    print("Get more data by augment")
    for root, dirs, files in os.walk(dataset_path):
        for obj_label in label:
            obj_list = [i for i in dirs if obj_label in i]
            if not obj_list:
                continue
            for obj in obj_list:
                # print("start seq_augmentation")
                extract_path = os.path.join(dataset_path, obj , 'light_mask')
                # print(extract_path)
                seq_augmentation.aug_frames(extract_path, aug_count[obj_label])


# Split to train and test by random
def split_to_train_test():
    print("Split dataset to train and test")

    global dataset_path
    global label

    num_label_list = dict.fromkeys(label, 0)
    print(num_label_list)
    seq_list = defaultdict(list)
    # input()

    for file in glob.glob('./sort_dataset/*/*light_mask/'):
        for obj_label in label :
            if obj_label in file:
                seq_list[obj_label].append(file)
                num_label_list[obj_label] += 1

    for obj_label in label :
        if not num_label_list[obj_label] :
            num_label_list.pop(obj_label, None)
    print(num_label_list)


    if os.path.exists('train'):
        shutil.rmtree('train')
    os.makedirs('train')

    if os.path.exists('test'):
        shutil.rmtree('test')
    os.makedirs('test')


    TEST_SEQ_COUNT=10


    for key, value in num_label_list.items():
        print(key + " : " + str(value))

        rr_seed = np.empty(10)
        for i in range(TEST_SEQ_COUNT):
            rr_seed[i] = int(random.random()*10000)
        print(rr_seed)

        test_obj_rand_set = []
        for rand in rr_seed :
            test_obj_rand = rand % value
            test_obj_rand_set.append(test_obj_rand)
        print(test_obj_rand_set )

        for i in range(value):
            if i in test_obj_rand_set:
                target_seq_dir = 'test'
                print( seq_list[key][i] +' is '+ target_seq_dir)
            else:
                target_seq_dir = 'train'
            file_split_dir = seq_list[key][i].split('\\')
            new_file_name = file_split_dir[-2]+'_'+file_split_dir[-3]
            if file_split_dir[-2] == 'light_mask':
                new_file_name = new_file_name+'/light_mask'
            # print(new_file_name)
            shutil.copytree(seq_list[key][i], os.path.join(target_seq_dir, new_file_name))

if __name__ == '__main__':
    print(args.aug)
    aug_count = counting_label()
    do_seq_augmentation(aug_count, args.aug)
    # split_to_train_test()