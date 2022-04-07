Before run this folder's .py, make soure already done the get_label_img_from_xml_to_file.py in folder /taillight_image_dataset
step1.
	run sort_dataset_with_label.py
	it will go throught ./taillight_image_dataset/result_data
	that will create a new folder /sort_dataset and it sort not follow mp4 name but label of the images
step2.
	run split_dataset_to_train_test.py
	it must to have seq_augmentation.py in the same folder
	it do three things 
		+ Counting every label need to augment 
		if --aug True
		    + Do seq_augmentation, add augment data in /sort_dataset
		+ Split to train and test by random
	after those finished, there are two folder /train /test created
step3.
	run trans_dir_to_label.py [dir of target folder]
	collect the same label data under target folder
	it will create the label_folder which has data in source folder
