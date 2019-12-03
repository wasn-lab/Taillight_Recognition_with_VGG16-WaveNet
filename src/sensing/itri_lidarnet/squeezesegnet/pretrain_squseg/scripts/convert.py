"""
author: Quack 
date: Feb 20th, 2019

Converting from .npy to .txt 
	python convert.py --inpath ../data/npyFile/ \
	                 			 --outpath ../data/testFile_out/ \
					  	 --conv txt

Converting from .txt to .npy
python convert.py --inpath /home/babyduck/nn_ws/lidar_detector/src/lidar_squseg_detect/data/pre_hino/txtFile/ --outpath /home/babyduck/nn_ws/lidar_detector/src/lidar_squseg_detect/data/pre_hino/ --outdir npyFile --conv npy

"""
from tqdm import tqdm
from statistics import mode
import numpy as np
import time as t
import itertools
import argparse
import sys
import os

def makeDir(path, dirN):
	"""
	Create output directory. Provide path, directory name, and ID to indicate if it was 
	labeled with / without ground. 
		e.g. g_v --> ground, ng_v --> non-ground. 
	The method will create a new version of the directory if it already exists.  
	"""
	ver = 0
	newDir = dirN
	fullPath = path + newDir
	while os.path.isdir(fullPath):
		print "Directory {} exists.".format(fullPath)
		ver += 1
		newDir = dirN + str(ver)
		fullPath = path + newDir
	print "Creating directory {}".format(fullPath)
	os.makedirs(fullPath)
	return fullPath

def arrSize(file, viewType):
	"""
	Output the azimuth according to the given view-type and filename
	"""

	azimuth = 512    # default
	if viewType == "T":
		phi_center = ["N135","P0","P135"]
		imageWidth = [512,1024,512] 

		for ind in range(len(phi_center)):
			if phi_center[ind] in file:
				azimuth = imageWidth[ind]
	elif viewType == "X":
		azimuth = 512

	return azimuth


def npyConvert(ipath, opath, zenith, viewType):
	"""
	Convert point cloud files in .txt format to .npy format to be processed by the SqueezeSeg
	network. Provide input path where the .txt files are and full path of directory created to 
	save the .npy files. The method will create and .npy files with shape (64, 512, 6) as 
	indicated in the SqueezeSeg approach. 
	"""
	numFiles = len(os.listdir(ipath))
	params = 6  # x, y, z, i, r, l

	print "\nProcessing  {:} .txt files from: {} ".format(numFiles, ipath)
	with tqdm(total=numFiles) as progressBar:
		for file in os.listdir(ipath):
			with open(ipath + file, 'r') as f:
				azimuth = arrSize(file,viewType)
				npyArr = np.zeros((zenith, azimuth, params))
			 	txtArr = np.loadtxt(ipath + file)
			 	# Convert .txt into a (64, 512, 6) np.array
			 	currentMin = 0
				currentMax = azimuth
			 	for i in range(zenith):
			 		npyArr[i, :, :] = txtArr[currentMin:currentMax, :]
			 		currentMin = currentMax
			 		currentMax += azimuth
			# Save to .npy file 
			npyFilename = '/' + file.split('.')[0] + '.npy'
			np.save(opath + npyFilename, npyArr)
			progressBar.update(1)

def txtConvert(ipath, opath):
	"""
	Convert point cloud files in .npy format to .txt format to be processed and annotated by the 
	Ground Filtration Algorithm. Provide input path where the .npy files are and full path of 
	directory created to save the .txt files. 
	"""
	numFiles = len(os.listdir(ipath))	
	print "\nProcessing {:} .npy files from: {}".format(numFiles, ipath)
	with tqdm(total=numFiles) as progressBar:
		for file in os.listdir(ipath):
			# Read current point cloud and create file 
			pointCloudPath = ipath + file
			pointCloud = np.load(pointCloudPath).astype(np.float32)
			txtFilename = '/' + file.split('.')[0] + '.txt'
			outfile = open(opath + txtFilename, 'a')

			# Each line has a format [X Y Z I R L]
			for i, j in itertools.product(range(pointCloud.shape[0]), range(pointCloud.shape[1])):
				xyzirl = pointCloud[i, j, :] 
				pointCloud[i,j,3] = 0
				xyzirl_str = ' '.join(str(k) for k in xyzirl) + '\n'
				
				outfile.write(xyzirl_str)
			outfile.close()
			progressBar.update(1)

def main():
	# Argument parsing 
	ap = argparse.ArgumentParser()
	ap.add_argument("--inpath",  required=True,   help="Path to .npy files")
	ap.add_argument("--outpath", required=True,   help="Path to .txt files")
	ap.add_argument("--conv",    required=True,   help="'npy' or 'txt': type of conversion")
	ap.add_argument("--outdir",  required=True,   help="Name of output directory.")
	ap.add_argument("--zenith",  default=64,      help="Number of point cloud rings")
	ap.add_argument("--viewtype",default="X",     help="Viewtype for partition of phi")
	args = vars(ap.parse_args())

	inPath   = args["inpath"]
	outPath  = args["outpath"]
	zenith   = args["zenith"]
	outDir   = args["outdir"]
	convType = args["conv"]
	viewType = args["viewtype"]

	# Process data
	start = t.time() # Program start
	if convType == "npy": 
		npyConvert(inPath, makeDir(outPath, outDir), zenith, viewType)
	elif convType == "txt":
		txtConvert(inPath, makeDir(outPath, outDir))
	else:
		print "Error: invalid argument"
		exit()
	print "Execution Time: ", t.time() - start, "s"

if __name__ == '__main__':
	main()
