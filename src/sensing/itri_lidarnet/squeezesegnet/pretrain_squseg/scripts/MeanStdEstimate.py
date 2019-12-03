from tqdm import tqdm
from statistics import mode
import numpy as np
import time as t
import itertools
import argparse
import sys
import os

def calcuMean(npyArr,params):
	meanArr = np.zeros((1,params))
	for ind in range(params):
		meanArr[0,ind] = np.mean(npyArr[:,:,ind])

	return meanArr

def calcuStd(npyArr,params):
	stdArr = np.zeros((1,params))
	for ind in range(params):
		stdArr[0,ind] = np.std(npyArr[:,:,ind])
	
	return stdArr

def calcuLabelCnt(npyArr,labels):
	labelArr = np.zeros((1,labels))
	npy_select = npyArr[npyArr[:,:,4]>0]
	for ind in range(labels):
		labelArr[0,ind] = np.sum(npy_select[:,5] == ind)
	
	return labelArr

def main():
	# Argument parsing 
	ap = argparse.ArgumentParser()
	ap.add_argument("--inpath",  required=True)
	ap.add_argument("--viewtype", required=True)
	# ap.add_argument("--outpath", required=True)
	args = vars(ap.parse_args())

	inPath   = args["inpath"]
	viewType = args["viewtype"]
	# outPath  = args["outpath"]

	if viewType=="X":
		phi_center = ["N90","P0","P90","P180"]
	elif  viewType=="T":
		phi_center = ["P0","P135","N135"]
	else:
		print("No matched viewtype found")

	params = 5    # 5: x,y,z.i,d
	labels = 4    # 4: unknown, car, pedestrian, cyclist

	# Process data
	numFiles = len(os.listdir(inPath))

	print(numFiles)

	meanArr = np.zeros((numFiles/len(phi_center),len(phi_center),params))
	stdArr = np.zeros((numFiles/len(phi_center),len(phi_center),params))
	labelArr = np.zeros((numFiles/len(phi_center),len(phi_center),labels))

	fileind = np.zeros((1,len(phi_center))).astype(np.int) #  ["P0deg","P90deg","P180deg","N90deg"]

	with tqdm(total=numFiles) as progressBar:
		for file in os.listdir(inPath):
			pointCloudPath = inPath + file
			pointCloud = np.load(pointCloudPath).astype(np.float32)
			# print(pointCloud.shape)
			for phi_center_ind in range(len(phi_center)):
				if phi_center[phi_center_ind] in file:
					meanArr[fileind[0,phi_center_ind],phi_center_ind,:] = calcuMean(pointCloud,params)
					stdArr[fileind[0,phi_center_ind],phi_center_ind,:] = calcuStd(pointCloud,params)
					labelArr[fileind[0,phi_center_ind],phi_center_ind,:] = calcuLabelCnt(pointCloud,labels)
					fileind[0,phi_center_ind] = fileind[0,phi_center_ind]+1
			progressBar.update(1)
		print("Mean: ")
		print(np.mean(meanArr,axis=0))
		print(np.mean(meanArr,axis=0).shape)
		print("Std: ")
		print(np.mean(stdArr,axis=0))
		print(np.mean(stdArr,axis=0).shape)
		print("Class cnt: ")
		print(np.mean(labelArr,axis=0))
		print(np.mean(labelArr,axis=0).shape)

		labelArr_nor = np.mean(labelArr,axis=0)
		for phi_center_ind in range(len(phi_center)):
			labelArr_nor[phi_center_ind,:] = labelArr_nor[phi_center_ind,:]/labelArr_nor[phi_center_ind,1]
		print("Class cnt (normalized): ")
		print(labelArr_nor)
		print(labelArr_nor.shape)

		for phi_center_ind in range(len(phi_center)):
			labelArr_nor[phi_center_ind,:] = 1/labelArr_nor[phi_center_ind,:]
		print("Class cnt (normalized reciprocal): ")
		print(labelArr_nor)
		print(labelArr_nor.shape)

		
if __name__ == '__main__':
	main()
