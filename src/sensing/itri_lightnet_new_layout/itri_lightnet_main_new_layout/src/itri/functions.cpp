#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <fstream>
#include <iterator>
#include <string>
#include <math.h>
#include "functions.h"
#include <opencv2/highgui/highgui.hpp>  //for reading and visualizing image
#include <opencv2/imgproc/imgproc.hpp>  //for image resizing

using namespace cv;
using namespace std;

int falseCount = 0;
int preDepth;

//Data-type Converter Subroutine
string intToString(int number)
{
	stringstream ss;
	ss << number;
	return ss.str();
}
string floatToString(float number)
{
	ostringstream ss;
	ss << number;
	return ss.str();
}

int tClass, tClass2;
Point tStart, tStop, tStart2, tStop2;
float IOU;
int optimumIDX;
int maxVal;
bool inconsistentStatus30 = false, inconsistentDetails30[6];		//USER_DEFINED_ARRAY
bool inconsistentStatus60 = false, inconsistentDetails60[6];		//USER_DEFINED_ARRAY

float averageS, averageI, averageP;
float tempS, tempI, tempP;
float R = 0.0, B = 0.0, G = 0.0, S = 0.0, I = 0.0;
float aux_RG = 0.0, aux_RB = 0.0, aux_GB = 0.0;
float aux_c1 = 0.0, aux_c2 = 0.0;
float theta, aux_hue;
int colorHist[2];

int imageWidth  = 608; 	//USER_DEFINED_PARAM
int imageHeight = 384;	//USER_DEFINED_PARAM
int depthFOV = 50;	 	//USER_DEFINED_PARAM
int cnvFrame_th = 6; 	//USER_DEFINED_PARAM
int minDims = 3;   		//USER_DEFINED_PARAM

float min_ch(float R, float G, float B)
{
	float min = R;
	if (G<min) { min = G; }
	if (B<min) { min = B; }
	return min;
}

float max_ch(float R, float G, float B)
{
	float max = 0;
	if (R > max) 
	{
		max = R;
	}
	if (G > max) 
	{
		max = G;
	}
	if (B > max) 
	{
		max = B;
	}
	return max;
}

void adaptiveThreshold(int Body[9], Mat RGB, float &threshI, float &threshS)
{
	I = 0;
	for (int j = Body[2]; j < Body[4]; j++)
	{
		for (int i = Body[1]; i < Body[3]; i++)
		{
			B = (float)(RGB.at<Vec3b>(j, i)[0]) / 255;
			G = (float)(RGB.at<Vec3b>(j, i)[1]) / 255;
			R = (float)(RGB.at<Vec3b>(j, i)[2]) / 255;

			I += max_ch(R, G, B);

		}
	}
	threshI = (I / (float)(Body[5])) * 1.1;
			
	S = 0;
	for (int j = Body[2]; j < Body[4]; j++)
	{
		for (int i = Body[1]; i < Body[3]; i++)
		{
			B = (float)(RGB.at<Vec3b>(j, i)[0]) / 255;
			G = (float)(RGB.at<Vec3b>(j, i)[1]) / 255;
			R = (float)(RGB.at<Vec3b>(j, i)[2]) / 255;

			aux_c1 = R + G + B;
			aux_c2 = min_ch(R, G, B);
			if (aux_c1 != 0)
			{
				S += (1 - ((3 / aux_c1) * aux_c2));
			}
		}
	}
	threshS = (S / (float)(Body[5])) * 0.1;
}

int dimAspectRatio(int inpBulb[8])
{
	int width, height;
	width  = inpBulb[3] - inpBulb[1];
	height = inpBulb[4] - inpBulb[2];
	if (width > height)
	{
		return 1;
	} 
	else
	{
		return 2;
	}
}

float intersectionOverUnion(int box1[8], int box2[8])
{
	float minx1 = (float)box1[1];
	float maxx1 = (float)box1[3];
	float miny1 = (float)box1[2];
	float maxy1 = (float)box1[4];
	float minx2 = (float)box2[1];
	float maxx2 = (float)box2[3];
	float miny2 = (float)box2[2];
	float maxy2 = (float)box2[4];

	if (minx1 > maxx2 || maxx1 < minx2 || miny1 > maxy2 || maxy1 < miny2 || box1[0] != box2[0])
	{
		return 0.0f;
	}
	else
	{
		float dx = std::min(maxx2, maxx1) - std::max(minx2, minx1);
		float dy = std::min(maxy2, maxy1) - std::max(miny2, miny1);
		float area1 = (maxx1 - minx1) * (maxy1 - miny1);
		float area2 = (maxx2 - minx2) * (maxy2 - miny2);
		float inter = dx * dy;
		float uni = area1 + area2 - inter; 
		float IoU = inter / uni;
		return (IoU);
	}
}

void filterBB (int inpBulb[90][8], int &numInp, int outBulb[90][10], int &numOut, int camSelector)
{
	int width, width_2, height, height_2;
	//1. Remove Distant Bulbs based on Pixel-based Dimensions
	for (int l = 0; l < numInp; l++)
	{
		width  = inpBulb[l][3] - inpBulb[l][1];
		height = inpBulb[l][4] - inpBulb[l][2];
		if (width > height)
		{
			if (width < minDims)
			{
				inpBulb[l][0] = 0;
			}
		} 
		else
		{
			if (height < minDims)
			{
				inpBulb[l][0] = 0;
			}
		}
	}
	//2. Remove Overlapping Bulbs
	int simOverClass[2]; float simOverIOU = 0;
	int difOverClass[2]; float difOverIOU = 0;
	difOverClass[0] = 0; difOverClass[1] = 0;
	for (int l = 0; l < numInp; l++)
	{
		if (inpBulb[l][0] > 0)
		{
			for (int k = 0; k < numInp; k++)
			{
				if (l != k && inpBulb[k][0] > 0)
				{
					IOU = intersectionOverUnion(inpBulb[l], inpBulb[k]);
					if (inpBulb[k][6] == inpBulb[l][6])
					{
						if (IOU > simOverIOU)
						{
							simOverClass[0] = k;
							simOverClass[1] = l;
							simOverIOU = IOU;
						}
					}
					else
					{
						if (IOU > difOverIOU)
						{
							difOverClass[0] = k;
							difOverClass[1] = l;
							difOverIOU = IOU;
						}
					}
				}
			}
		}
	}
	int ARScore[2];
	if (simOverIOU > 0)
	{
		for (int j = 0; j < 2; j++)
		{
			width  = inpBulb[simOverClass[j]][3] - inpBulb[simOverClass[j]][1];
			height = inpBulb[simOverClass[j]][4] - inpBulb[simOverClass[j]][2];
			ARScore[j] = abs(width - height);
		}
		if (ARScore[0] > ARScore[1])
		{
			inpBulb[simOverClass[0]][0] = 0;
		}
		else
		{
			inpBulb[simOverClass[1]][0] = 0;
		}
	}
	if (difOverIOU > 0)
	{
		if (inpBulb[difOverClass[0]][0] > 0 && inpBulb[difOverClass[1]][0])
		{
			tStart.x  = inpBulb[difOverClass[0]][1];
			tStart.y  = inpBulb[difOverClass[0]][2];
			tStop.x   = inpBulb[difOverClass[0]][3];
			tStop.y   = inpBulb[difOverClass[0]][4];
			tStart2.x  = inpBulb[difOverClass[1]][1];
			tStart2.y  = inpBulb[difOverClass[1]][2];
			tStop2.x   = inpBulb[difOverClass[1]][3];
			tStop2.y   = inpBulb[difOverClass[1]][4];
			int diffX[2], diffY[2];
			int deltaX, deltaY;
			diffX[0] = abs(tStop.x - tStart2.x);
			diffX[1] = abs(tStart.x - tStop2.x);
			diffY[0] = abs(tStop.y - tStart2.y);
			diffY[1] = abs(tStart.y - tStop2.y);
			if (diffX[0] < diffX[1])
			{
				deltaX = diffX[1];
			}
			else
			{
				deltaX = diffX[0];
			}
			if (diffY[0] < diffY[1])
			{
				deltaY = diffY[1];
			}
			else
			{
				deltaY = diffY[0];
			}
			if (deltaX > deltaY)
			{
				//Fix X Dimension
				width   = tStop.x - tStart.x;
				width_2 = tStop2.x - tStart2.x;
				if (tStart.x < tStart2.x)
				{
					if (tStop.x < tStop2.x)
					{
						//Case 1
						if (width > width_2)
						{
							inpBulb[difOverClass[0]][3] = tStart2.x - 1;
						}
						else
						{
							inpBulb[difOverClass[1]][1] = tStop.x + 1;
						}
					}
					else
					{
						//Case 3
						if ((width - width_2) < (width / 2))
						{
							if ((tStart2.x - tStart.x) > (tStop.x - tStop2.x))
							{
								inpBulb[difOverClass[0]][1] = tStart.x;
								inpBulb[difOverClass[0]][3] = tStart.x + (width / 2);
								inpBulb[difOverClass[1]][1] = inpBulb[difOverClass[0]][3] + 1;
								inpBulb[difOverClass[1]][3] = tStop.x;
							}
							else
							{
								inpBulb[difOverClass[1]][1] = tStart.x;
								inpBulb[difOverClass[1]][3] = tStart.x + (width / 2);
								inpBulb[difOverClass[0]][1] = inpBulb[difOverClass[1]][3] + 1;
								inpBulb[difOverClass[0]][3] = tStop.x;	
							}
						}
						else
						{
							inpBulb[difOverClass[1]][0] = 0;
						}
					}
				}
				else
				{
					if (tStop.x > tStop2.x)
					{
						//Case 2
						if (width > width_2)
						{
							inpBulb[difOverClass[0]][1] = tStop2.x + 1;
						}
						else
						{
							inpBulb[difOverClass[1]][3] = tStart.x - 1;
						}
					}
					else
					{
						//Case 4
						if ((width_2 - width) < (width_2 / 2))
						{
							if ((tStart.x - tStart2.x) > (tStop2.x - tStop.x))
							{
								inpBulb[difOverClass[0]][1] = tStart2.x;
								inpBulb[difOverClass[0]][3] = tStart2.x + (width_2 / 2);
								inpBulb[difOverClass[1]][1] = inpBulb[difOverClass[0]][3] + 1;
								inpBulb[difOverClass[1]][3] = tStop2.x;
							}
							else
							{
								inpBulb[difOverClass[1]][1] = tStart2.x;
								inpBulb[difOverClass[1]][3] = tStart2.x + (width_2 / 2);
								inpBulb[difOverClass[0]][1] = inpBulb[difOverClass[1]][3] + 1;
								inpBulb[difOverClass[0]][3] = tStop2.x;	
							}
						}
						else
						{
							inpBulb[difOverClass[0]][0] = 0;
						}
					}
				}
			}
			else
			{
				//Fix Y Dimension
				height   = tStop.y - tStart.y;
				height_2 = tStop2.y - tStart2.y;
				if (tStart.y < tStart2.y)
				{
					if (tStop.y < tStop2.y)
					{
						//Case 1
						if (height > height_2)
						{
							inpBulb[difOverClass[0]][4] = tStart2.y - 1;
						}
						else
						{
							inpBulb[difOverClass[1]][2] = tStop.y + 1;
						}
					}
					else
					{
						//Case 3
						if ((height - height_2) < (height / 2))
						{
							if ((tStart2.y - tStart.y) > (tStop.y - tStop2.y))
							{
								inpBulb[difOverClass[0]][2] = tStart.y;
								inpBulb[difOverClass[0]][4] = tStart.y + (height / 2);
								inpBulb[difOverClass[1]][2] = inpBulb[difOverClass[0]][4] + 1;
								inpBulb[difOverClass[1]][4] = tStop.y;
							}
							else
							{
								inpBulb[difOverClass[1]][2] = tStart.y;
								inpBulb[difOverClass[1]][4] = tStart.y + (height / 2);
								inpBulb[difOverClass[0]][2] = inpBulb[difOverClass[1]][4] + 1;
								inpBulb[difOverClass[0]][4] = tStop.y;	
							}
						}
						else
						{
							inpBulb[difOverClass[1]][0] = 0;
						}
					}
				}
				else
				{
					if (tStop.y > tStop2.y)
					{
						//Case 2
						if (height > height_2)
						{
							inpBulb[difOverClass[0]][2] = tStop2.y + 1;
						}
						else
						{
							inpBulb[difOverClass[1]][4] = tStart.y - 1;
						}
					}
					else
					{
						//Case 4
						if ((height_2 - height) < (height_2 / 2))
						{
							if ((tStart.y - tStart2.y) > (tStop2.y - tStop.y))
							{
								inpBulb[difOverClass[0]][2] = tStart2.y;
								inpBulb[difOverClass[0]][4] = tStart2.y + (height_2 / 2);
								inpBulb[difOverClass[1]][2] = inpBulb[difOverClass[0]][4] + 1;
								inpBulb[difOverClass[1]][4] = tStop2.y;
							}
							else
							{
								inpBulb[difOverClass[1]][2] = tStart2.y;
								inpBulb[difOverClass[1]][4] = tStart2.y + (height_2 / 2);
								inpBulb[difOverClass[0]][2] = inpBulb[difOverClass[1]][4] + 1;
								inpBulb[difOverClass[0]][4] = tStop2.y;	
							}
						}
						else
						{
							inpBulb[difOverClass[0]][0] = 0;
						}
					}
				}
			}
		}
	}

	//3. Estimating Depth Value for Each BB based on Camera Selector Identifier
	numOut = 0;
	for (int l = 0; l < numInp; l++)
	{
		if (inpBulb[l][0] > 0)
		{
			outBulb[numOut][0] = inpBulb[l][0];
			outBulb[numOut][1] = inpBulb[l][1];
			outBulb[numOut][2] = inpBulb[l][2];
			outBulb[numOut][3] = inpBulb[l][3];
			outBulb[numOut][4] = inpBulb[l][4];
			outBulb[numOut][5] = (inpBulb[l][3] - inpBulb[l][1]) * (inpBulb[l][4] * inpBulb[l][2]);
			outBulb[numOut][6] = inpBulb[l][6];
			outBulb[numOut][7] = -1;
			outBulb[numOut][8] = -1;
			if (camSelector == 30)
			{
				float f_30 = 1111;
				outBulb[numOut][9] = ((f_30 * 27) / float(outBulb[numOut][4] - outBulb[numOut][2])) / 100;
			}
			else if (camSelector == 60)
			{		
				float f_60 = 555.5;
				if(float(outBulb[numOut][4] - outBulb[numOut][2]) == 0)
					outBulb[numOut][9] = 0;
				else
					outBulb[numOut][9] = ((f_60 * 27) / float(outBulb[numOut][4] - outBulb[numOut][2])) / 100;
			}
			numOut++;
		}
	}
}

void clusterBB (int Bulb[90][10], int numBB, int Box[90][10], int &numBox)
{
	int numCluster = 0;
	int countCluster[90];
	for (int k = 0; k < 90; k++)
	{
		countCluster[k] = 0;
	}

	//1. Clustering BB based on Assumption of Horizontal TL
	for (int l = 0; l < numBB; l++)
	{
		if (Bulb[l][8] == -1)
		{
			tStart.x = Bulb[l][1] + ((Bulb[l][3] - Bulb[l][1]) / 2);
			tStart.y = Bulb[l][2] + ((Bulb[l][4] - Bulb[l][2]) / 2);
			Bulb[l][8] = numCluster;
			countCluster[Bulb[l][8]]++;
			for (int m = 0; m < numBB; m++)
			{
				if (l != m)
				{
					tStart2.x = Bulb[m][1] + ((Bulb[m][3] - Bulb[m][1]) / 2);
					tStart2.y = Bulb[m][2] + ((Bulb[m][4] - Bulb[m][2]) / 2);
					float distX = 0.75 * (tStart2.x - tStart.x);
					float distY = 1.25 * (tStart2.y - tStart.y);
					float dist = sqrt((distX * distX) + (distY * distY));
					if (dist < 30)
					{
						if (Bulb[m][8] == -1)
						{
							Bulb[m][8] = Bulb[l][8];
							countCluster[Bulb[m][8]]++;
						}
					}
				}
			}
			numCluster++;
		}
	}

	//2. Clustering BB based on Assumption of Vertical TL
	for (int k = 0; k < numCluster; k++)
	{
		if (countCluster[k] == 1)
		{
			for(int l = 0; l < numBB; l++)
			{
				if (Bulb[l][8] == k)
				{
					tStart.x = Bulb[l][1] + ((Bulb[l][3] - Bulb[l][1]) / 2);
					tStart.y = Bulb[l][2] + ((Bulb[l][4] - Bulb[l][2]) / 2);
					for (int m = 0; m < numBB; m++)
					{
						if (l != m && countCluster[Bulb[m][8]] == 1)
						{
							tStart2.x = Bulb[m][1] + ((Bulb[m][3] - Bulb[m][1]) / 2);
							tStart2.y = Bulb[m][2] + ((Bulb[m][4] - Bulb[m][2]) / 2);
							float distX = 1.25 * (tStart2.x - tStart.x);
							float distY = 0.75 * (tStart2.y - tStart.y);
							float dist = sqrt((distX * distX) + (distY * distY));
							if (dist < 30)
							{
								countCluster[Bulb[m][8]]--;
								Bulb[m][8] = Bulb[l][8];
								countCluster[Bulb[m][8]]++;
							}
						}
					}
				}
			}
		}
	}

	//3. Arranging Cluster Information
	int localizeBB[90];
	int depthValue;
	for (int l = 0; l < numBB; l++)
	{
		localizeBB[l] = Bulb[l][8];
	}
	numBox = 0;
	for (int k = 0; k < numCluster; k++)
	{
		if (countCluster[k] > 0)
		{
			int insideCount = 0;
			depthValue = 0;
			for(int l = 0; l < numBB; l++)
			{
				if (localizeBB[l] == k)
				{
					Bulb[l][8] = numBox;
					Box[numBox][insideCount + 2] = l;
					insideCount++;
					Box[numBox][0] = insideCount;
					depthValue += Bulb[l][9];
				}
			}
			if (Box[numBox][0] > 0)
			{
				Box[numBox][1] = depthValue / Box[numBox][0];	
			}
			else
			{
				Box[numBox][1] = -1;
			}
			numBox++;
		}
	}
}

	//Information for _tlBulb
	//[0] = Availability Status
	//[1] = X Start
	//[2] = Y Start
	//[3] = X Stop
	//[4] = Y Stop
	//[5] = Dimension
	//[6] = Class
	//[7] = Verified Class
	//[8] = TL Box Position
	//[9] = Depth Information

	//Information for _tlBox
	//[0] = Number of Bulb
	//[1] = Depth
	//[2-9] = 8 Slots for Traffic Light

void statusRecognize30(int Bulb[90][10], int numBB, int Box[90][10], int numBox, int &selectedBox,
					 Mat RGB, bool firstFrame, int preCount[6], bool finStat[6])
{
	//1. Localizing Nearest Set of TL Bulbs from Current Frame
	Point center;
	float distX, distY, bbDist, confDist;
	distX = (float)imageWidth  / 2;
	distY = (float)imageHeight / 8;
	float baseDist = sqrt((distX * distX) + (distY * distY));
	float bestDist = 0;
	selectedBox = -1;

	bool nowStat[6], preStat[6];
	int  nowStatNum, preStatNum;
	//1.B Has Connection or Not?
	bool hasSimilarityToPrevious;
	int similarBoxToPrevious[30], numSimilarBox = 0;
	preStatNum = 0;
	for (int s = 0; s < 6; s++)
	{
		if (preCount[s] > 0)
		{
			preStat[s] = true;
			preStatNum++;
		}
		else
		{
			preStat[s] = false;
		}
	}
	for (int k = 0; k < numBox; k++)
	{
		hasSimilarityToPrevious = true;
		for (int s = 0; s < 6; s++)
		{
			nowStat[s] = false;
		}
		nowStatNum = 0;
		for (int l = 0; l < Box[k][0]; l++)
		{
			nowStat[Bulb[Box[k][l + 2]][6]] = true;
			nowStatNum++;
		}
		if (nowStatNum < preStatNum)
		{
			for (int s = 0; s < 6; s++)
			{
				if (nowStat[s] != preStat[s])
				{
					hasSimilarityToPrevious = false;
				}
			}	
		}
		if (hasSimilarityToPrevious)
		{
			similarBoxToPrevious[numSimilarBox] = k;
			numSimilarBox++;
		}
	}

	float maxCluster = 0;
	for (int k = 0; k < numBox; k++)
	{
		if (Box[k][0] > maxCluster)
		{
			maxCluster = Box[k][0];
		}
	}

	if (numSimilarBox == 0 || firstFrame)
	{
		for (int k = 0; k < numBox; k++)
		{
			center.x = 0; center.y = 0;
			for (int l = 0; l < Box[k][0]; l++)
			{
				center.x += ((Bulb[Box[k][l+2]][1] + Bulb[Box[k][l+2]][3]) / 2);
				center.y += ((Bulb[Box[k][l+2]][2] + Bulb[Box[k][l+2]][4]) / 2);
			}
			center.x /= Box[k][0];
			center.y /= Box[k][0];
			distX = ((float)center.x - ((float)imageWidth  / 2));
			distY = ((float)center.y - ((float)imageHeight / 8));
			bbDist = sqrt((distX * distX) + (distY * distY));
			confDist = 1 - (bbDist / baseDist);
			if (Box[k][1] < depthFOV && Box[k][0] == maxCluster)
			{
				if (confDist > bestDist)
				{
					selectedBox = k;
					bestDist = confDist;
				}
			}
		}
	}
	else
	{
		for (int k = 0; k < numSimilarBox; k++)
		{
			center.x = 0; center.y = 0;
			for (int l = 0; l < Box[similarBoxToPrevious[k]][0]; l++)
			{
				center.x += ((Bulb[Box[similarBoxToPrevious[k]][l+2]][1] + Bulb[Box[similarBoxToPrevious[k]][l+2]][3]) / 2);
				center.y += ((Bulb[Box[similarBoxToPrevious[k]][l+2]][2] + Bulb[Box[similarBoxToPrevious[k]][l+2]][4]) / 2);
			}
			center.x /= Box[similarBoxToPrevious[k]][0];
			center.y /= Box[similarBoxToPrevious[k]][0];
			distX = ((float)center.x - ((float)imageWidth  / 2));
			distY = ((float)center.y - ((float)imageHeight / 8));
			bbDist = sqrt((distX * distX) + (distY * distY));
			confDist = 1 - (bbDist / baseDist);
			if (confDist > bestDist && Box[similarBoxToPrevious[k]][1] < depthFOV)
			{
				selectedBox = similarBoxToPrevious[k];
				bestDist = confDist;
			}
		}
	}


	//2. Verifing Each TL Bulb of Selected TL
	if (selectedBox > -1)
	{
		//TL Bulb Verification
		for (int k = 0; k < Box[selectedBox][0]; k++)
		{
			for (int color = 0; color < 2; color++)
			{
				colorHist[color] = 0;
			}
			for (int j = Bulb[Box[selectedBox][k + 2]][2]; j < Bulb[Box[selectedBox][k + 2]][4]; j++)
			{
				for (int i = Bulb[Box[selectedBox][k + 2]][1]; i < Bulb[Box[selectedBox][k + 2]][3]; i++)
				{
					B = (float)(RGB.at<Vec3b>(j, i)[0]) / 255;
					G = (float)(RGB.at<Vec3b>(j, i)[1]) / 255;
					R = (float)(RGB.at<Vec3b>(j, i)[2]) / 255;
					aux_c1 = R + G + B;
					I = aux_c1 / 3;
					if (I > 0.0) 
					{
						aux_c2 = min_ch(R, G, B);
						S = 1 - ((3 / aux_c1)*aux_c2);
					}
					aux_c1 = 0.0; aux_c2 = 0.0;
					if (S > averageS) 
					{
						I = max_ch(R, G, B);
						if (I > averageI) 
						{
							aux_RG = R - G; aux_RB = R - B; aux_GB = G - B;
							aux_c1 = (aux_RG + aux_RB) / 2;
							aux_c2 = sqrt((aux_RG*aux_RG) + (aux_RB*aux_GB));
							theta = acos(aux_c1 / aux_c2);

							if (B <= G) 
							{ 
								aux_hue = theta * 57.1366246;
							}
							else 
							{ 
								aux_hue = ((2 * PI) - theta) * 57.1366246; 
							}

							if (aux_hue < 60)
							{
								colorHist[0]++;
							}
							else if (aux_hue > 120 && aux_hue < 220)
							{
								colorHist[1]++;
							}
							else if (aux_hue > 320)
							{
								colorHist[0]++;
							}
						}
					}
				}
			}
			maxVal = 0; optimumIDX = -1;
			for (int color = 0; color < 2; color++)
			{
				if (colorHist[color] > maxVal)
				{
					maxVal = colorHist[color];
					optimumIDX = color;
				}
			}

			//RED SPECTRUM
			if (optimumIDX == 0)
			{
				Bulb[Box[selectedBox][k + 2]][7] = 0;
			}
			//GREEN SPECTRUM
			else
			{
				Bulb[Box[selectedBox][k + 2]][7] = 2;
			}
		}
	}

	//3. Organizing Traffic Light
	for (int s = 0; s < 6; s++)
	{
		nowStat[s] = false;
		preStat[s] = false;
	}
	if (selectedBox > -1)
	{
		for (int k = 0; k < Box[selectedBox][0]; k++)
		{
			nowStat[Bulb[Box[selectedBox][k + 2]][6]] = true;
		}		
	}
	if (!firstFrame)
	{
		bool different = false;
		for (int s = 0; s < 6; s++)
		{
			if (preCount[s] > 0)
			{
				preStat[s] = true;
			}
			finStat[s] = nowStat[s];
			if (preStat[s] != nowStat[s])
			{
				different = true;
			}
		}
		//cout << inconsistentStatus30 << " " << different << " | " << preCount[0] << " " << preCount[1] << " " << preCount[2] << " " << preCount[3] << " " << preCount[4] << " " << preCount[5] << " | ";
		if (different)
		{
			if (!inconsistentStatus30)
			{
				for (int s = 0; s < 6; s++)
				{
					finStat[s] = preStat[s];
					if (finStat[s])
					{
						preCount[s]++;
					}
					inconsistentDetails30[s] = nowStat[s];
				}
				inconsistentStatus30 = true;
			}
			else
			{
				bool same = true;
				for (int s = 0; s < 6; s++)
				{
					if (nowStat[s] != inconsistentDetails30[s])
					{
						same = false;
					}
				}
				if (same)
				{
					for (int s = 0; s < 6; s++)
					{
						finStat[s] = nowStat[s];
						preCount[s] = 0;
						if (finStat[s])
						{
							preCount[s]++;
						}
					}
					inconsistentStatus30 = false;
				}
				else
				{
					for (int s = 0; s < 6; s++)
					{
						finStat[s] = preStat[s];
						if (finStat[s])
						{
							preCount[s]++;
						}
						inconsistentDetails30[s] = nowStat[s];
					}
				}
			}
		}
		else
		{
			inconsistentStatus30 = false;
			for (int s = 0; s < 6; s++)
			{
				finStat[s] = nowStat[s];
				if (finStat[s])
				{
					preCount[s]++;
				}
			}
		}
	}
	else
	{
		for (int s = 0; s < 6; s++)
		{
			finStat[s] = nowStat[s];
			if (finStat[s])
			{
				preCount[s]++;
			}
		}
	}
}

void statusRecognize60(int Bulb[90][10], int numBB, int Box[90][10], int numBox, int &selectedBox,
					 Mat RGB, bool firstFrame, int preCount[6], bool finStat[6])
{
	//1. Localizing Nearest Set of TL Bulbs from Current Frame
	Point center;
	float distX, distY, bbDist, confDist;
	distX = (float)imageWidth  / 2;
	distY = (float)imageHeight / 8;
	float baseDist = sqrt((distX * distX) + (distY * distY));
	float bestDist = 0;
	selectedBox = -1;

	bool nowStat[6], preStat[6];
	int  nowStatNum, preStatNum;
	//1.B Has Connection or Not?
	bool hasSimilarityToPrevious;
	int similarBoxToPrevious[30], numSimilarBox = 0;
	preStatNum = 0;
	for (int s = 0; s < 6; s++)
	{
		if (preCount[s] > 0)
		{
			preStat[s] = true;
			preStatNum++;
		}
		else
		{
			preStat[s] = false;
		}
	}
	for (int k = 0; k < numBox; k++)
	{
		hasSimilarityToPrevious = true;
		for (int s = 0; s < 6; s++)
		{
			nowStat[s] = false;
		}
		nowStatNum = 0;
		for (int l = 0; l < Box[k][0]; l++)
		{
			nowStat[Bulb[Box[k][l + 2]][6]] = true;
			nowStatNum++;
		}
		if (nowStatNum < preStatNum)
		{
			for (int s = 0; s < 6; s++)
			{
				if (nowStat[s] != preStat[s])
				{
					hasSimilarityToPrevious = false;
				}
			}	
		}
		if (hasSimilarityToPrevious)
		{
			similarBoxToPrevious[numSimilarBox] = k;
			numSimilarBox++;
		}
	}

	float maxCluster = 0;
	for (int k = 0; k < numBox; k++)
	{
		if (Box[k][0] > maxCluster)
		{
			maxCluster = Box[k][0];
		}
	}

	if (numSimilarBox == 0 || firstFrame)
	{
		for (int k = 0; k < numBox; k++)
		{
			center.x = 0; center.y = 0;
			for (int l = 0; l < Box[k][0]; l++)
			{
				center.x += ((Bulb[Box[k][l+2]][1] + Bulb[Box[k][l+2]][3]) / 2);
				center.y += ((Bulb[Box[k][l+2]][2] + Bulb[Box[k][l+2]][4]) / 2);
			}
			center.x /= Box[k][0];
			center.y /= Box[k][0];
			distX = ((float)center.x - ((float)imageWidth  / 2));
			distY = ((float)center.y - ((float)imageHeight / 8));
			bbDist = sqrt((distX * distX) + (distY * distY));
			confDist = 1 - (bbDist / baseDist);
			if (Box[k][1] < depthFOV && Box[k][0] == maxCluster)
			{
				if (confDist > bestDist)
				{
					selectedBox = k;
					bestDist = confDist;
				}		
			}
		}
	}
	else
	{
		for (int k = 0; k < numSimilarBox; k++)
		{
			center.x = 0; center.y = 0;
			for (int l = 0; l < Box[similarBoxToPrevious[k]][0]; l++)
			{
				center.x += ((Bulb[Box[similarBoxToPrevious[k]][l+2]][1] + Bulb[Box[similarBoxToPrevious[k]][l+2]][3]) / 2);
				center.y += ((Bulb[Box[similarBoxToPrevious[k]][l+2]][2] + Bulb[Box[similarBoxToPrevious[k]][l+2]][4]) / 2);
			}
			center.x /= Box[similarBoxToPrevious[k]][0];
			center.y /= Box[similarBoxToPrevious[k]][0];
			distX = ((float)center.x - ((float)imageWidth  / 2));
			distY = ((float)center.y - ((float)imageHeight / 8));
			bbDist = sqrt((distX * distX) + (distY * distY));
			confDist = 1 - (bbDist / baseDist);
			if (confDist > bestDist && Box[similarBoxToPrevious[k]][1] < depthFOV)
			{
				selectedBox = similarBoxToPrevious[k];
				bestDist = confDist;
			}
		}
	}


	//2. Verifing Each TL Bulb of Selected TL
	if (selectedBox > -1)
	{
		//TL Bulb Verification
		for (int k = 0; k < Box[selectedBox][0]; k++)
		{
			for (int color = 0; color < 2; color++)
			{
				colorHist[color] = 0;
			}
			for (int j = Bulb[Box[selectedBox][k + 2]][2]; j < Bulb[Box[selectedBox][k + 2]][4]; j++)
			{
				for (int i = Bulb[Box[selectedBox][k + 2]][1]; i < Bulb[Box[selectedBox][k + 2]][3]; i++)
				{
					B = (float)(RGB.at<Vec3b>(j, i)[0]) / 255;
					G = (float)(RGB.at<Vec3b>(j, i)[1]) / 255;
					R = (float)(RGB.at<Vec3b>(j, i)[2]) / 255;
					aux_c1 = R + G + B;
					I = aux_c1 / 3;
					if (I > 0.0) 
					{
						aux_c2 = min_ch(R, G, B);
						S = 1 - ((3 / aux_c1)*aux_c2);
					}
					aux_c1 = 0.0; aux_c2 = 0.0;
					if (S > averageS) 
					{
						I = max_ch(R, G, B);
						if (I > averageI) 
						{
							aux_RG = R - G; aux_RB = R - B; aux_GB = G - B;
							aux_c1 = (aux_RG + aux_RB) / 2;
							aux_c2 = sqrt((aux_RG*aux_RG) + (aux_RB*aux_GB));
							theta = acos(aux_c1 / aux_c2);

							if (B <= G) 
							{ 
								aux_hue = theta * 57.1366246;
							}
							else 
							{ 
								aux_hue = ((2 * PI) - theta) * 57.1366246; 
							}

							if (aux_hue < 60)
							{
								colorHist[0]++;
							}
							else if (aux_hue > 120 && aux_hue < 220)
							{
								colorHist[1]++;
							}
							else if (aux_hue > 320)
							{
								colorHist[0]++;
							}
						}
					}
				}
			}
			maxVal = 0; optimumIDX = -1;
			for (int color = 0; color < 2; color++)
			{
				if (colorHist[color] > maxVal)
				{
					maxVal = colorHist[color];
					optimumIDX = color;
				}
			}

			//RED SPECTRUM
			if (optimumIDX == 0)
			{
				Bulb[Box[selectedBox][k + 2]][7] = 0;
			}
			//GREEN SPECTRUM
			else
			{
				Bulb[Box[selectedBox][k + 2]][7] = 2;
			}
		}
	}

	//3. Organizing Traffic Light
	for (int s = 0; s < 6; s++)
	{
		nowStat[s] = false;
		preStat[s] = false;
	}
	if (selectedBox > -1)
	{
		for (int k = 0; k < Box[selectedBox][0]; k++)
		{
			nowStat[Bulb[Box[selectedBox][k + 2]][6]] = true;
		}		
	}
	if (!firstFrame)
	{
		bool different = false;
		for (int s = 0; s < 6; s++)
		{
			if (preCount[s] > 0)
			{
				preStat[s] = true;
			}
			finStat[s] = nowStat[s];
			if (preStat[s] != nowStat[s])
			{
				different = true;
			}
		}
		//cout << inconsistentStatus60 << " " << different << " | " << preCount[0] << " " << preCount[1] << " " << preCount[2] << " " << preCount[3] << " " << preCount[4] << " " << preCount[5] << " | ";
		if (different)
		{
			if (!inconsistentStatus60)
			{
				for (int s = 0; s < 6; s++)
				{
					finStat[s] = preStat[s];
					if (finStat[s])
					{
						preCount[s]++;
					}
					inconsistentDetails60[s] = nowStat[s];
				}
				inconsistentStatus60 = true;
			}
			else
			{
				bool same = true;
				for (int s = 0; s < 6; s++)
				{
					if (nowStat[s] != inconsistentDetails60[s])
					{
						same = false;
					}
				}
				if (same)
				{
					for (int s = 0; s < 6; s++)
					{
						finStat[s] = nowStat[s];
						preCount[s] = 0;
						if (finStat[s])
						{
							preCount[s]++;
						}
					}
					inconsistentStatus60 = false;
				}
				else
				{
					for (int s = 0; s < 6; s++)
					{
						finStat[s] = preStat[s];
						if (finStat[s])
						{
							preCount[s]++;
						}
						inconsistentDetails60[s] = nowStat[s];
					}
				}
			}
		}
		else
		{
			inconsistentStatus60 = false;
			for (int s = 0; s < 6; s++)
			{
				finStat[s] = nowStat[s];
				if (finStat[s])
				{
					preCount[s]++;
				}
			}
		}
	}
	else
	{
		for (int s = 0; s < 6; s++)
		{
			finStat[s] = nowStat[s];
			if (finStat[s])
			{
				preCount[s]++;
			}
		}
	}
}


void animateTrafficLight(Mat animate, bool finStat[6], int depthTL)
{
	if (depthTL > 0)
	{
		putText(animate, intToString(depthTL) + "m", Point(605, 85), 1, 7, Scalar(255, 255, 255), 5, 8, 0);
	}
	else
	{
		putText(animate, "N/A", Point(605, 85), 1, 7, Scalar(255, 255, 255), 5, 8, 0);
	}
	circle(animate, Point(50, 50), 50, Scalar(100, 100, 100), 2, 8, 0);
	circle(animate, Point(150, 50), 50, Scalar(100, 100, 100), 2, 8, 0);
	circle(animate, Point(250, 50), 50, Scalar(100, 100, 100), 2, 8, 0);
	circle(animate, Point(350, 50), 50, Scalar(100, 100, 100), 2, 8, 0);
	circle(animate, Point(450, 50), 50, Scalar(100, 100, 100), 2, 8, 0);
	circle(animate, Point(550, 50), 50, Scalar(100, 100, 100), 2, 8, 0);
	if (finStat[0] == true)
	{
		circle(animate, Point(50, 50), 50, Scalar(0, 0, 255), -1, 8, 0);
	}
	if (finStat[1] == true)
	{
		circle(animate, Point(150, 50), 50, Scalar(0, 255, 255), -1, 8, 0);
	}
	if (finStat[2] == true)
	{
		circle(animate, Point(250, 50), 50, Scalar(0, 255, 0), -1, 8, 0);
	}
	if (finStat[5] == true)
	{
		arrowedLine(animate, Point(395,50), Point(305, 50), Scalar(0, 255, 0), 10, 8, 0, 0.7);
	}
	if (finStat[3] == true)
	{
		arrowedLine(animate, Point(450,95), Point(450, 5), Scalar(0, 255, 0), 10, 8, 0, 0.7);
	}
	if (finStat[4] == true)
	{
		arrowedLine(animate, Point(505,50), Point(595, 50), Scalar(0, 255, 0), 10, 8, 0, 0.7);
	}
}

void finalizeTrafficLight(bool StatA[6], int depthA, bool StatB[6], int depthB, bool finStat[2][6], int finDepth[2])
{
	preDepth = finDepth[1];
	int aveDepth;
	if (preDepth > 0)
	{
		aveDepth = (preDepth + depthA + depthB)/3;
	}
	else
	{
		aveDepth = (depthA + depthB)/2;
	}
	if (abs(aveDepth - depthA) < abs(aveDepth - depthB))
	{
		if (depthA > 0)
		{
			finDepth[1] = depthA;
			for (int k = 0; k < 6; k++)
			{
				finStat[1][k] = StatA[k];
			}
		}
		else
		{
			finDepth[1] = depthB;
			for (int k = 0; k < 6; k++)
			{
				finStat[1][k] = StatB[k];
			}
		}
	}
	else
	{
		if (depthB > 0)
		{
			finDepth[1] = depthB;
			for (int k = 0; k < 6; k++)
			{
				finStat[1][k] = StatB[k];
			}
		}
		else
		{
			finDepth[1] = depthA;
			for (int k = 0; k < 6; k++)
			{
				finStat[1][k] = StatA[k];
			}
		}
	}

	if (finDepth[1] > 0)
	{
		if (finDepth[0] > 0)
		{
			bool matchFrame = true;
			for (int k = 0; k < 6; k++)
			{
				if (finStat[0][k] != finStat[1][k])
				{
					matchFrame = false;
				}
			}
			if (matchFrame == false)
			{
				bool matchcurLight = true;
				for (int k = 0; k < 6; k++)
				{
					if (finStat[1][k] != StatA[k])
					{
						matchcurLight = false;
					}
					if (finStat[1][k] != StatB[k])
					{
						matchcurLight = false;
					}
				}
				if (matchcurLight == false)
				{
					bool matchpreLight;
					bool matchA = true, matchB = true;
					bool emptyA = true, emptyB = true;
					for (int k = 0; k < 6; k++)
					{
						if (finStat[0][k] != StatA[k])
						{
							matchA = false;
						}
						if (finStat[0][k] != StatB[k])
						{
							matchB = false;
						}
						if (StatA[k] == true)
						{
							emptyA = false;
						}
						if (StatB[k] == true)
						{
							emptyB = false;
						}
					}
					matchpreLight = matchA | matchB;
					if (matchpreLight == false)
					{
						if (falseCount < 3)
						{
							for (int k = 0; k < 6; k++)
							{
								finStat[1][k] = finStat[0][k];
							}
							falseCount++;
						}
						else
						{
							falseCount = 0;
						}
						
					}
					else
					{
						if (matchA == true)
						{
							if (emptyA == true)
							{
								for (int k = 0; k < 6; k++)
								{
									finStat[1][k] = StatB[k];
								}
							}
							else
							{
								for (int k = 0; k < 6; k++)
								{
									finStat[1][k] = StatA[k];
								}
							}
						}
						if (matchB == true)
						{
							if (emptyB == true)
							{
								for (int k = 0; k < 6; k++)
								{
									finStat[1][k] = StatA[k];
								}
							}
							else
							{
								for (int k = 0; k < 6; k++)
								{
									finStat[1][k] = StatB[k];
								}
							}
						}
					}
				}
			}
		}
	}
	else
	{
		if (depthA == 0 && depthB == 0)
		{
			//do nothing
		}
		else
		{
			if (finDepth[0] > 0)
			{
				bool matchA = true, matchB = true;
				for (int k = 0; k < 6; k++)
				{
					if (finStat[0][k] != StatA[k])
					{
						matchA = false;
					}
					if (finStat[0][k] != StatB[k])
					{
						matchB = false;
					}
				}
				if (matchA == true)
				{
					for (int k = 0; k < 6; k++)
					{
						finStat[1][k] = StatA[k];
					} finDepth[1] = depthA;
				}
				if (matchB == true)
				{
					for (int k = 0; k < 6; k++)
					{
						finStat[1][k] = StatB[k];
					} finDepth[1] = depthB;
				}
			}
		}
	}

	for(int k = 0; k < 6; k++)
	{
		finStat[0][k] = finStat[1][k];
	} 
	finDepth[0] = finDepth[1];
}
