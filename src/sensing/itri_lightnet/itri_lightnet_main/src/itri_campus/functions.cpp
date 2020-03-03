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
float IOU, IOA, IOA2, bestIOA;
int insecArea, unionArea;
int temp_var, tempIDX;
int optimumIDX;

int minDimension, maxDimension, normDimension, possibleBulb, bulbDimension;

Point scanStart, scanStop;
float averageS, averageI, averageP;
float tempS, tempI, tempP;
float R = 0.0, B = 0.0, G = 0.0, S = 0.0, I = 0.0;
float aux_RG = 0.0, aux_RB = 0.0, aux_GB = 0.0;
float aux_c1 = 0.0, aux_c2 = 0.0;
float theta, aux_hue;
int colorHist[3];

int selector;

float distance, min_distance;
float auxMath1, auxMath2;

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

void filterTrafficLight(int inp_tlBody[30][6], int num_tlBody, int inp_tlBulb[90][8], int num_tlBulb, 
						int out_tlBody[30][7], int out_tlBulb[90][9],
						Point dimensions)
{

	//0. Resetting the Variables:
	for (int l = 0; l < 30; l++)
	{
		for (int m = 0; m < 7; m++)
		{
			out_tlBody[l][m] = -1;
		}
	}

	//1. Filter the traffic light body result based on overlapping condition.
	if (num_tlBody > 1)
	{
		for (int l = 0; l < num_tlBody; l++)
		{
			if (inp_tlBody[l][0] == 1)
			{
				tStart.x = inp_tlBody[l][1];
				tStart.y = inp_tlBody[l][2];
				tStop.x = inp_tlBody[l][3];
				tStop.y = inp_tlBody[l][4];
				for (int k = 0; k < num_tlBody; k++)
				{
					if (k != l && inp_tlBody[k][0] == 1)
					{
						insecArea = 0;
						tStart2.x = inp_tlBody[k][1];
						tStart2.y = inp_tlBody[k][2];
						tStop2.x = inp_tlBody[k][3];
						tStop2.y = inp_tlBody[k][4];
						for (int j = tStart.y; j < tStop.y; j++)
						{
							if (j > tStart2.y && j < tStop2.y)
							{
								for (int i = tStart.x; i < tStop.x; i++)
								{
									if (i > tStart2.x && i < tStop2.x)
									{
										insecArea++;
									}
								}
							}
						}
						unionArea = inp_tlBody[l][5] + inp_tlBody[k][5] - insecArea;
						IOU = (float)(insecArea) / (float)(unionArea);
						if (IOU >  0.8)
						{
							if (tStart.x > tStart2.x)
							{
								tStart.x = tStart2.x;
							}
							if (tStart.y > tStart2.y)
							{
								tStart.y = tStart2.y;
							}
							if (tStop.x < tStop2.x)
							{
								tStop.x = tStop2.x;
							}
							if (tStop.y < tStop2.y)
							{
								tStop.y = tStop2.y;
							}
							inp_tlBody[l][1] = tStart.x;
							inp_tlBody[l][2] = tStart.y;
							inp_tlBody[l][3] = tStop.x;
							inp_tlBody[l][4] = tStop.y;
							inp_tlBody[l][5] = (tStop.x - tStart.x) * (tStop.y - tStart.y);
							inp_tlBody[k][0] = 0;
						}
						else
						{
							if (inp_tlBody[l][5] > inp_tlBody[k][5])
							{
								IOA = (float)(insecArea) / (float)(inp_tlBody[k][5]);
								if (IOA > 0.8)
								{
									inp_tlBody[k][0] = 0;
								}
							}
							else
							{
								IOA = (float)(insecArea) / (float)(inp_tlBody[l][5]);
								if (IOA > 0.8)
								{
									inp_tlBody[l][0] = 0;
								}
							}
						}
					}
					else {}
				}
			}
			else {}
		}
	}
	else {}

	//2. Remove any traffic light bulb outlier.
	if (num_tlBulb > 0)
	{
		minDimension = dimensions.x * dimensions.y;
		maxDimension = 0;
		for (int k = 0; k < num_tlBulb; k++)
		{
			if (minDimension > inp_tlBulb[k][5])
			{
				minDimension = inp_tlBulb[k][5];
			}
			if (maxDimension < inp_tlBulb[k][5])
			{
				maxDimension = inp_tlBulb[k][5];
			}
		}
		minDimension = minDimension * 1.5;
		maxDimension = maxDimension * 6.5;
	}
	else
	{
		minDimension = dimensions.x * dimensions.y * 0.00005; //changed
		maxDimension = dimensions.x * dimensions.y * 0.01000; //changed
	}

	temp_var = 0;
	for (int l = 0; l < num_tlBody; l++)
	{
		if (inp_tlBody[l][0] == 1)
		{
			if (inp_tlBody[l][5] > minDimension && inp_tlBody[l][5] < maxDimension)
			{
				out_tlBody[temp_var][0] = 0;
				for (int m = 1; m < 6; m++)
				{
					out_tlBody[temp_var][m] = inp_tlBody[l][m];
				}
				out_tlBody[temp_var][6] = int(round((134.5*384*360/40.5/float(out_tlBody[temp_var][4]-out_tlBody[temp_var][2])/6.283/100)));
				temp_var++;
			}
		}
	}
	num_tlBody = temp_var;

	//3. Filter the traffic light bulb result based on overlapping condition.
	if (num_tlBulb > 1)
	{
		for (int k = 0; k < num_tlBulb; k++)
		{
			if (inp_tlBulb[k][0] == 1)
			{
				tStart.x = inp_tlBulb[k][1];
				tStart.y = inp_tlBulb[k][2];
				tStop.x = inp_tlBulb[k][3];
				tStop.y = inp_tlBulb[k][4];
				tClass = inp_tlBulb[k][6];
				for (int l = 0; l < num_tlBulb; l++)
				{
					if (k != l && inp_tlBulb[l][0] == 1)
					{
						tClass2 = inp_tlBulb[l][6];
						insecArea = 0;
						tStart2.x = inp_tlBulb[l][1];
						tStart2.y = inp_tlBulb[l][2];
						tStop2.x = inp_tlBulb[l][3];
						tStop2.y = inp_tlBulb[l][4];
						for (int j = tStart.y; j < tStop.y; j++)
						{
							if (j > tStart2.y && j < tStop2.y)
							{
								for (int i = tStart.x; i < tStop.x; i++)
								{
									if (i > tStart2.x && i < tStop2.x)
									{
										insecArea++;
									}
								}
							}
						}
						unionArea = inp_tlBulb[k][5] + inp_tlBulb[l][5] - insecArea;
						IOU = (float)(insecArea) / (float)(unionArea);
						if (IOU >  0.8)
						{
							if (tClass == tClass2)
							{
								if (tStart.x > tStart2.x)
								{
									tStart.x = tStart2.x;
								}
								if (tStart.y > tStart2.y)
								{
									tStart.y = tStart2.y;
								}
								if (tStop.x < tStop2.x)
								{
									tStop.x = tStop2.x;
								}
								if (tStop.y < tStop2.y)
								{
									tStop.y = tStop2.y;
								}
								inp_tlBulb[k][1] = tStart.x;
								inp_tlBulb[k][2] = tStart.y;
								inp_tlBulb[k][3] = tStop.x;
								inp_tlBulb[k][4] = tStop.y;
								inp_tlBulb[k][5] = (tStop.x - tStart.x) * (tStop.y - tStart.y);
								inp_tlBulb[l][0] = 0;
							}
							else
							{
								if (inp_tlBulb[k][5] > inp_tlBulb[l][5])
								{
									inp_tlBulb[l][0] = 0;
								}
								else
								{
									inp_tlBulb[k][0] = 0;
								}
							}
						}
						else
						{
							if (inp_tlBulb[k][5] > inp_tlBulb[l][5])
							{
								IOA = (float)(insecArea) / (float)(inp_tlBulb[l][5]);
								if (IOA > 0.8)
								{
									inp_tlBulb[l][0] = 0;
								}
							}
							else
							{
								IOA = (float)(insecArea) / (float)(inp_tlBulb[k][5]);
								if (IOA > 0.8)
								{
									inp_tlBulb[k][0] = 0;
								}
							}
						}
					}
				}
			}
		}
	}

	//4B. Identifying the ROI for each traffic light bulb.
	temp_var = 0;
	for (int k = 0; k < num_tlBulb; k++)
	{
		if (inp_tlBulb[k][0] == 1)
		{
			optimumIDX = -1;
			bestIOA = 0;
			tStart.x = inp_tlBulb[k][1];
			tStart.y = inp_tlBulb[k][2];
			tStop.x = inp_tlBulb[k][3];
			tStop.y = inp_tlBulb[k][4];
			for (int l = 0; l < num_tlBody; l++)
			{
				insecArea = 0;
				tStart2.x = out_tlBody[l][1];
				tStart2.y = out_tlBody[l][2];
				tStop2.x = out_tlBody[l][3];
				tStop2.y = out_tlBody[l][4];
				for (int j = tStart.y; j < tStop.y; j++)
				{
					if (j > tStart2.y && j < tStop2.y)
					{
						for (int i = tStart.x; i < tStop.x; i++)
						{
							if (i > tStart2.x && i < tStop2.x)
							{
								insecArea++;
							}
						}
					}
				}
				IOA = (float)(insecArea) / (float)(inp_tlBulb[k][5]);
				if (IOA > bestIOA)
				{
					bestIOA = IOA;
					optimumIDX = l;
				}
			}
			if (optimumIDX > -1 && bestIOA > 0.3)
			{
				out_tlBody[optimumIDX][0] = 1;
				for (int m = 0; m < 7; m++)
				{
					out_tlBulb[temp_var][m] = inp_tlBulb[k][m];
				}
				out_tlBulb[temp_var][7] = -1;
				out_tlBulb[temp_var][8] = optimumIDX;
			}
			else
			{
				for (int m = 0; m < 7; m++)
				{
					out_tlBulb[temp_var][m] = inp_tlBulb[k][m];
				}
				out_tlBulb[temp_var][7] = -1;
				out_tlBulb[temp_var][8] = -1;
			}
			temp_var++;
		}
	}
	num_tlBulb = temp_var;
}

void adaptiveThreshold(int Body[7], Mat RGB, float &threshI, float &threshS)
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
			S += (1 - ((3 / aux_c1) * aux_c2));
		}
	}
	threshS = (S / (float)(Body[5])) * 1.1;
}

int checkOrientation(int Body[7], int &minDim, int &maxDim)
{
	int width  	 = Body[3] - Body[1];
	int height 	 = Body[4] - Body[2];
	int shape = -1;
	if (width > height)
	{
		shape = 1;
		minDim = height;
		maxDim = width;
	}
	else
	{
		shape = 2;
		minDim = width;
		maxDim = height;
	}
	return shape;
}

void proposeTrafficLight(int tlBody[30][7], int num_tlBody, Mat RGB, int tlBulb[90][9], int &num_tlBulb)
{
	int scanPoint, scanRange[2], bbLimit[2], scanResolution;
	int pixelCondition;

	int tlHist[6][2];
	int tlPixl[6];
	int tlStat[6];
	bool lightStat = false;

	possibleBulb = 5; //This setup is constrained to only 5 Traffic Light

	for (int l = 0; l < num_tlBody; l++)
	{
		if (tlBody[l][0] == 0)
		{

			lightStat = false;

			selector = checkOrientation(tlBody[l], minDimension, maxDimension);
			adaptiveThreshold(tlBody[l], RGB, averageI, averageS);
			temp_var = minDimension / 2;
			if (selector == 1)
			{
				scanPoint = tlBody[l][2] + temp_var;
				bbLimit[0] = tlBody[l][2];
				bbLimit[1] = tlBody[l][4];
				scanRange[0] = tlBody[l][1];
				scanRange[1] = tlBody[l][3];
			}
			else
			{
				scanPoint = tlBody[l][1] + temp_var;
				bbLimit[0] = tlBody[l][1];
				bbLimit[1] = tlBody[l][3];
				scanRange[0] = tlBody[l][2];
				scanRange[1] = tlBody[l][4];
			}
			scanResolution = (scanRange[1] - scanRange[0]) / possibleBulb;

			for (int k = 0; k < possibleBulb; k++)
			{
				tlPixl[k] = 0;
				for (int c = 0; c < 2; c++)
				{
					tlHist[k][c] = 0;
				}
			} pixelCondition = -1;

			int x, y;
			for (int k = 0; k < possibleBulb; k++)
			{
				for (int i = scanRange[0] + (k * scanResolution); i < scanRange[0] + ((k + 1) * scanResolution); i++)
				{
					for (int j = scanPoint; j < scanPoint + 2; j++)
					{
						if (selector == 1)
						{
							x = i;
							y = j;
						}
						else
						{
							x = j;
							y = i;
						}
						B = (float)(RGB.at<Vec3b>(y, x)[0]) / 255;
						G = (float)(RGB.at<Vec3b>(y, x)[1]) / 255;
						R = (float)(RGB.at<Vec3b>(y, x)[2]) / 255;

						aux_c1 = R + G + B;
						I = max_ch(R, G, B);
						//Checking the Pixel Intensity Level
						if (I >= averageI)
						{
							aux_c2 = min_ch(R, G, B);
							//Checking the Pixel Saturation Level
							if (S >= averageS)
							{
								aux_RG = R - G; aux_RB = R - B; aux_GB = G - B;
								aux_c1 = (aux_RG + aux_RB) / 2;
								aux_c2 = sqrt((aux_RG * aux_RG) + (aux_RB * aux_GB));
								theta = acos(aux_c1 / aux_c2);
								if (B <= G) 
								{ 
									aux_hue = theta * 57.1366246;
								}
								else 
								{ 
									aux_hue = ((2 * PI) - theta) * 57.1366246; 
								}
								//Checking the Pixel Spectrum Position
								if (aux_hue < 60)
								{
									//RED
									pixelCondition = 0;
								}
								else if (aux_hue > 150 && aux_hue < 190)
								{
									//GREEN
									pixelCondition = 1;
								}
								else if (aux_hue > 340)
								{
									//RED
									pixelCondition = 0;
								}
								else
								{
									//OTHER COLORS
									pixelCondition = -1;
								}
							}
							else
							{
								pixelCondition = -1;
							}

							if (pixelCondition > -1)
							{
								tlPixl[k]++;
								tlHist[k][pixelCondition]++;
							}
						}

						

					}
				}
			}

			for (int k = 0; k < 2; k++)
			{
				if (tlPixl[k] > 0)
				{
					if (tlHist[k][0] > tlHist[k][1])
					{
						tlStat[k] = 1;
					}
					else
					{
						tlStat[k] = 0;
					}
				}
				else
				{
					tlStat[k] = 0;
				}
			}
			if (tlStat[0] == tlStat[1] && tlStat[0] == 1)
			{
				if (tlHist[0][0] >= tlHist[1][0])
				{
					tlStat[1] = 0;
				}
				else
				{
					tlStat[0] = 0;
				}
			}

			for (int k = 2; k < possibleBulb; k++)
			{
				if (tlPixl[k] > 0)
				{
					if (tlHist[k][0] > tlHist[k][1])
					{
						tlStat[k] = 0;
					}
					else
					{
						tlStat[k] = 1;
						lightStat = true;
					}
				}
				else
				{
					tlStat[k] = 0;
				}
			}
			if (lightStat == true && tlStat[0] == 1)
			{
				for (int k = 2; k < possibleBulb; k++)
				{
					tlStat[k] = 0;
				}
			}

			num_tlBulb = 0;
			for (int k = 0; k < possibleBulb; k++)
			{
				if (tlStat[k] == 1)
				{
					tlBulb[num_tlBulb][0] = 1;
					if (selector == 1)
					{
						tlBulb[num_tlBulb][1] = scanRange[0] + (k * scanResolution);
						tlBulb[num_tlBulb][2] = bbLimit[0];
						tlBulb[num_tlBulb][3] = scanRange[0] + ((k + 1) * scanResolution);
						tlBulb[num_tlBulb][4] = bbLimit[1];
					}
					else
					{
						tlBulb[num_tlBulb][1] = bbLimit[0];
						tlBulb[num_tlBulb][2] = scanRange[0] + (k * scanResolution);
						tlBulb[num_tlBulb][3] = bbLimit[1];
						tlBulb[num_tlBulb][4] = scanRange[0] + ((k + 1) * scanResolution);
					}
					tlBulb[num_tlBulb][5] = (tlBulb[num_tlBulb][4] - tlBulb[num_tlBulb][2]) * (tlBulb[num_tlBulb][3] - tlBulb[num_tlBulb][1]);
					tlBulb[num_tlBulb][6] = -1;
					tlBulb[num_tlBulb][7] = k;
					tlBulb[num_tlBulb][8] = l;
					num_tlBulb++;
				}
			}

		}
	}
}