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
float IOU, IOA, bestIOA;
int insecArea, unionArea;
int temp_var, tempIDX;
int optimumIDX;

int minDimension, maxDimension, normDimension, possibleBulb, bulbDimension;
int cnvCounter = 0;

Point scanStart, scanStop;
float averageS, averageI, averageP;
float tempS, tempI, tempP;
float R = 0.0, B = 0.0, G = 0.0, S = 0.0, I = 0.0;
float aux_RG = 0.0, aux_RB = 0.0, aux_GB = 0.0;
float aux_c1 = 0.0, aux_c2 = 0.0;
float theta, aux_hue;
int colorHist[2];
int selector;

int depthFOV = 40;	 //USER_DEFINED_PARAM
int dPixel_th = 3; 	 //USER_DEFINED_PARAM
int bulb_th = 3;     //USER_DEFINED_PARAM
int cnvFrame_th = 3; //USER_DEFINED_PARAM

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

void filterTrafficLight(int inp_tlBody[30][6], int &num_tlBody, int inp_tlBulb[90][8], int &num_tlBulb, 
						int out_tlBody[30][9], int out_tlBulb[90][9],
						int &selectBody, Point dimensions)
{

	//0. Resetting the Variables:
	for (int l = 0; l < 30; l++)
	{
		for (int m = 0; m < 8; m++)
		{
			out_tlBody[l][m] = -1;
		}
		out_tlBody[l][8] = 0;
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
						for (int j = tStart.y; j <= tStop.y; j++)
						{
							if (j >= tStart2.y && j <= tStop2.y)
							{
								for (int i = tStart.x; i <= tStop.x; i++)
								{
									if (i >= tStart2.x && i <= tStop2.x)
									{
										insecArea++;
									}
								}
							}
						}
						unionArea = inp_tlBody[l][5] + inp_tlBody[k][5] - (2 * insecArea);
						IOU = (float)(insecArea) / (float)(unionArea);
						if (IOU >  0.5)
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
								if (IOA > 0.5)
								{
									inp_tlBody[k][0] = 0;
								}
							}
							else
							{
								IOA = (float)(insecArea) / (float)(inp_tlBody[l][5]);
								if (IOA > 0.5)
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

	//2. Remove any traffic light body outlier.
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
		minDimension = minDimension * 1;

	}
	else
	{
		minDimension = dimensions.x * dimensions.y * 0.0001;
	}
	maxDimension = dimensions.x * dimensions.y * 0.050;

	temp_var = 0;
	for (int l = 0; l < num_tlBody; l++)
	{
		if (inp_tlBody[l][0] == 1)
		{
			if (inp_tlBody[l][5] >= minDimension && inp_tlBody[l][5] < maxDimension)
			{
				out_tlBody[temp_var][0] = 0;
				for (int m = 1; m < 6; m++)
				{
					out_tlBody[temp_var][m] = inp_tlBody[l][m];
				}
				out_tlBody[temp_var][6] = int(round((36*384*360/25.5/float(out_tlBody[temp_var][4]-out_tlBody[temp_var][2])/6.283/100)));
				out_tlBody[temp_var][7] = -1;
				temp_var++;
			}
		}
	}
	num_tlBody = temp_var;

	//3. Filter the traffic light bulb result based on overlapping condition.
	//cout << "NUM BULB: " << num_tlBulb << endl;
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
						for (int j = tStart.y; j <= tStop.y; j++)
						{
							if (j >= tStart2.y && j <= tStop2.y)
							{
								for (int i = tStart.x; i <= tStop.x; i++)
								{
									if (i >= tStart2.x && i <= tStop2.x)
									{
										insecArea++;
									}
								}
							}
						}
						unionArea = inp_tlBulb[k][5] + inp_tlBulb[l][5] - (2 * insecArea);
						IOU = (float)(insecArea) / (float)(unionArea);

						//cout << IOU << " " << k << " " << l << endl;

						if (IOU >  0.5)
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
								if (IOA > 0.5)
								{
									inp_tlBulb[l][0] = 0;
								}
							}
							else
							{
								IOA = (float)(insecArea) / (float)(inp_tlBulb[k][5]);
								if (IOA > 0.5)
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
			//cout << "IOA " << optimumIDX << " " << bestIOA << endl;
			if (optimumIDX > -1 && bestIOA > 0.1)
			{
				out_tlBody[optimumIDX][0] = 1;
				out_tlBody[optimumIDX][8]++;
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

	bool emptyTrafficLightBody = true;
	for (int l = 0; l < num_tlBody; l++)
	{
		if (out_tlBody[l][8] > 0)
		{
			emptyTrafficLightBody = false;
		}
	}
	selectBody = -1;
	int depthMin = depthFOV;
	for (int l = 0; l < num_tlBody; l++)
	{
		if (emptyTrafficLightBody == true)
		{
			if (out_tlBody[l][6] < depthMin)
			{
				depthMin = out_tlBody[l][6];
				selectBody = l;
			}
		}
		else
		{
			if (out_tlBody[l][6] < depthMin && out_tlBody[l][8] > 0)
			{
				depthMin = out_tlBody[l][6];
				selectBody = l;
			}
		}	
	}
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

int checkOrientation(int Body[9], int &minDim, int &maxDim)
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

void improveTrafficLight(int tlBody[9], int selectBody, int inBulb[5], int num_tlBulb, int tlBulb[90][9], Mat RGB)
{
	int numSelectBulb = 0, class_info;
	for (int k = 0; k < 5; k++)
	{
		inBulb[k] = -1;
	}

	selector = checkOrientation(tlBody, minDimension, maxDimension);
	for (int k = 0; k < num_tlBulb; k++)
	{
		if (tlBulb[k][8] == selectBody)
		{
			inBulb[numSelectBulb] = k;

			averageI = 0; averageS = 0;
			//Improving the dimensionality of the traffic bulb based on traffic body reference.
			if (selector == 1)
			{
				if (tlBulb[k][1] < tlBody[1])
				{
					tlBulb[k][1] = tlBody[1];
				}
				if (tlBulb[k][3] > tlBody[3])
				{
					tlBulb[k][3] = tlBody[3];
				}
			}
			else
			{
				if (tlBulb[k][2] < tlBody[2])
				{
					tlBulb[k][2] = tlBody[2];
				}
				if (tlBulb[k][4] > tlBody[4])
				{
					tlBulb[k][4] = tlBody[4];
				}
			}

			class_info = tlBulb[k][6];
			if (class_info < 3)
			{
				minDimension = (tlBulb[k][selector + 2] - tlBulb[k][selector]) * 1.1;
			}
			else
			{
				minDimension = (tlBulb[k][selector + 2] - tlBulb[k][selector]);
			}
			if (tlBody[7] == -1)
			{
				tlBody[7] = minDimension;
			}
			else
			{
				bulbDimension = (tlBody[7] +  minDimension) / 2;
				tlBody[7] = bulbDimension;
			}

			numSelectBulb++;
		}
	}
}


void verifyTrafficLight(int tlBody[9], int inBulb[5], int tlBulb[90][9], Mat RGB)
{

	for (int k = 0; k < tlBody[8]; k++)
	{
		selector = checkOrientation(tlBody, minDimension, maxDimension);
		maxDimension *= 0.75;
		minDimension = tlBody[7];
		possibleBulb = int(round(float(maxDimension) / float(minDimension)));
		//cout << maxDimension << " " << float(maxDimension) / float(minDimension) << endl;
		if (possibleBulb < 3)
		{
			possibleBulb = 3;
			normDimension = maxDimension / 3;
		}
		else if (possibleBulb > 5)
		{
			possibleBulb = 5;
			normDimension = maxDimension / 5;
		}
		else
		{
			normDimension = maxDimension / possibleBulb;
		}
		tlBody[7] = possibleBulb;
		//cout << "POSSIBLE LIGHTS = " << possibleBulb << "  LIGHT SIZE = " << normDimension << endl;

		for (int color = 0; color < 2; color++)
		{
			colorHist[color] = 0;
		}
		//Verify the class information.
		for (int j = tlBulb[inBulb[k]][2]; j < tlBulb[inBulb[k]][4]; j++)
		{
			for (int i = tlBulb[inBulb[k]][1]; i < tlBulb[inBulb[k]][3]; i++)
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
							//RED
							colorHist[0]++;
						}
						else if (aux_hue > 120 && aux_hue < 220){
							//GREEN
							colorHist[1]++;
						}
						else if (aux_hue > 320)
						{
							//RED
							colorHist[0]++;
						}
					}
				}
			}
		}
		maxDimension = 0; optimumIDX = -1;
		for (int color = 0; color < 2; color++)
		{
			if (colorHist[color] > maxDimension)
			{
				maxDimension = colorHist[color];
				optimumIDX = color;
			}
		}

		//RED SPECTRUM
		if (optimumIDX == 0)
		{
			tempIDX = ceil((((float)(tlBulb[inBulb[k]][selector + 2] - tlBulb[inBulb[k]][selector]) / 2) + tlBulb[inBulb[k]][selector] - tlBody[selector]) / normDimension);
			if (tempIDX < 2)
			{
				tlBulb[inBulb[k]][7] = 0;
			}
			else
			{
				tlBulb[inBulb[k]][7] = 1;
			}
		}
		//GREEN SPECTRUM
		else
		{
			tlBulb[inBulb[k]][7] = 2;
		}
	
		/*string netResult, predictColor;
		if (tlBulb[inBulb[k]][6] == 0)
		{
			netResult = "FULL RED";
		}
		else if (tlBulb[inBulb[k]][6] == 1)
		{
			netResult = "FULL YELLOW";
		}
		else if (tlBulb[inBulb[k]][6] == 2)
		{
			netResult = "FULL GREEN";
		}
		else if (tlBulb[inBulb[k]][6] == 3)
		{
			netResult = "AHEAD GREEN";
		}
		else if (tlBulb[inBulb[k]][6] == 4)
		{
			netResult = "RIGHT GREEN";
		}	
		else if (tlBulb[inBulb[k]][6] == 5)
		{
			netResult = "LEFT GREEN";
		}
		else
		{
			netResult = "DIGIT RED";
		}



		if (tlBulb[inBulb[k]][7] == -1){
			predictColor = "UNKNOWN";
		}
		else if (tlBulb[inBulb[k]][7] == 0)
		{
			predictColor = "RED";
		}
		else if (tlBulb[inBulb[k]][7] == 1)
		{
			predictColor = "YELLOW";
		}
		else if (tlBulb[inBulb[k]][7] == 2)
		{
			predictColor = "GREEN";
		}
		else
		{
			predictColor = "UNKNOWN";
		}
		cout << "NET RESULT = " << netResult << "  VERIFIED AS = " << predictColor << endl;*/
	}
}


void findTrafficLight(int tlBody[9], int selectBody, int inBulb[5], int &num_tlBulb, int tlBulb[90][9], Mat RGB)
{
	int tolerance;
	int checkBox = 0, locBox, resBox, distBox, mindistBox, optimumBox = 0;
	int checkStatus[5][2];
	bool anyRedYellow = false, anyGreen = false;
	bool anyMultiGreen = false;
	int maxHist = 0, bestColor = -1;
	for (int l = 0; l < tlBody[8]; l++)
	{
		if (tlBulb[inBulb[l]][7] < 2)
		{
			anyRedYellow = true;
		}
		if (tlBulb[inBulb[l]][7] == 2)
		{
			if (tlBulb[inBulb[l]][6] == 2)
			{
				anyMultiGreen = true;
			}
			anyGreen = true;
			checkBox++;
		}
	}
	if (checkBox > 1)
	{
		anyMultiGreen = true;
	}
	checkBox = 0;
	
	adaptiveThreshold(tlBody, RGB, averageI, averageS);
	averageI = averageI * 1.5;
	averageS = averageS * 2;
	selector = checkOrientation(tlBody, minDimension, maxDimension);
	locBox = round((float)maxDimension / (float)tlBody[7]);
	resBox = round((float)locBox / 2);
	if (selector == 1)
	{
		for (int k = 0; k < tlBody[7]; k++)
		{
			scanStart.x = (locBox * k) + tlBody[1];
			checkStatus[k][0] = resBox + scanStart.x;
			checkStatus[k][1] = -1;
		}
		for (int l = 0; l < tlBody[8]; l++)
		{
			checkBox = round((float)(tlBulb[inBulb[l]][1] + tlBulb[inBulb[l]][3]) / 2);
			mindistBox = 9999;
			for (int k = 0; k < tlBody[7]; k++)
			{
				distBox = abs(checkBox - checkStatus[k][0]);
				if (distBox < mindistBox && checkStatus[k][1] == -1)
				{
					mindistBox = distBox;
					optimumBox = k;
				}
			}
			checkStatus[optimumBox][1] = l;
		}
		for (int k = 0; k < tlBody[7]; k++)
		{
			if (checkStatus[k][1] == -1)
			{
				tolerance = (int)round(((float)tlBody[4] - (float)tlBody[2]) / 3);
				if (tolerance < 1)
				{
					tolerance = 1;
				}
				scanStart.x = (locBox * k) + tlBody[1];
				scanStart.y = tlBody[2] + tolerance; 
				scanStop.x  = (locBox * (k + 1)) + tlBody[1];
				scanStop.y  = tlBody[4] - tolerance;
				for (int color = 0; color < 2; color++)
				{
					colorHist[color] = 0;
				}
				//Verify the class information.
				for (int j = scanStart.y; j < scanStop.y; j++)
				{
					for (int i = scanStart.x; i < scanStop.x; i++)
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
									//RED
									colorHist[0]++;
								}
								else if (aux_hue > 120 && aux_hue < 220){
									//GREEN
									colorHist[1]++;
								}
								else if (aux_hue > 320)
								{
									//RED
									colorHist[0]++;
								}
							}
						}
					}
				}

				maxHist = 0; bestColor = -1;
				for (int color = 0; color < 2; color++)
				{
					if (colorHist[color] > maxHist)
					{
						maxHist = colorHist[color];
						bestColor = color;
					}
				}

				if (bestColor > -1)
				{
					if (k < 2)
					{
						if (bestColor == 0 && anyMultiGreen == false)
						{
							if (anyRedYellow == false && maxHist > 5)
							{
								//add new bounding box.
								if (k == 0)
								{
									//cout << maxHist << " RED BB ADDED" << endl;
									tlBulb[num_tlBulb][0] = 2;
									tlBulb[num_tlBulb][1] = scanStart.x;
									tlBulb[num_tlBulb][2] = scanStart.y - tolerance;
									tlBulb[num_tlBulb][3] = scanStop.x;
									tlBulb[num_tlBulb][4] = scanStop.y + tolerance;
									tlBulb[num_tlBulb][5] = (tlBulb[num_tlBulb][4] - tlBulb[num_tlBulb][2]) * (tlBulb[num_tlBulb][3] - tlBulb[num_tlBulb][1]);
									tlBulb[num_tlBulb][6] = -1;
									tlBulb[num_tlBulb][7] = 0;
									tlBulb[num_tlBulb][8] = selectBody;
									inBulb[tlBody[8]] = num_tlBulb;
									tlBody[8]++;
									num_tlBulb++;
									anyRedYellow = true;
								}
								else
								{
									//cout << maxHist << " YELLOW BB ADDED" << endl;
									tlBulb[num_tlBulb][0] = 2;
									tlBulb[num_tlBulb][1] = scanStart.x;
									tlBulb[num_tlBulb][2] = scanStart.y - tolerance;
									tlBulb[num_tlBulb][3] = scanStop.x;
									tlBulb[num_tlBulb][4] = scanStop.y + tolerance;
									tlBulb[num_tlBulb][5] = (tlBulb[num_tlBulb][4] - tlBulb[num_tlBulb][2]) * (tlBulb[num_tlBulb][3] - tlBulb[num_tlBulb][1]);
									tlBulb[num_tlBulb][6] = -1;
									tlBulb[num_tlBulb][7] = 1;
									tlBulb[num_tlBulb][8] = selectBody;
									inBulb[tlBody[8]] = num_tlBulb;
									tlBody[8]++;
									num_tlBulb++;
									anyRedYellow = true;
								}
							}
						}
					}
					else
					{
						if (bestColor == 1)
						{
							if (anyMultiGreen == false && anyRedYellow == false)
							{
								if (anyGreen == true)
								{
									if (maxHist > 0)
									{
										//add new bounding box.
										//cout << maxHist << " GREEN BB ADDED" << endl;
										tlBulb[num_tlBulb][0] = 2;
										tlBulb[num_tlBulb][1] = scanStart.x;
										tlBulb[num_tlBulb][2] = scanStart.y - tolerance;
										tlBulb[num_tlBulb][3] = scanStop.x;
										tlBulb[num_tlBulb][4] = scanStop.y + tolerance;
										tlBulb[num_tlBulb][5] = (tlBulb[num_tlBulb][4] - tlBulb[num_tlBulb][2]) * (tlBulb[num_tlBulb][3] - tlBulb[num_tlBulb][1]);
										tlBulb[num_tlBulb][6] = -1;
										tlBulb[num_tlBulb][7] = 2;
										tlBulb[num_tlBulb][8] = selectBody;
										inBulb[tlBody[8]] = num_tlBulb;
										tlBody[8]++;
										num_tlBulb++;
										anyMultiGreen = true;
									}
								}
								else
								{
									if (maxHist > 10)
									{
										//add new bounding box.
										//cout << maxHist << " GREEN BB ADDED" << endl;
										tlBulb[num_tlBulb][0] = 2;
										tlBulb[num_tlBulb][1] = scanStart.x;
										tlBulb[num_tlBulb][2] = scanStart.y - tolerance;
										tlBulb[num_tlBulb][3] = scanStop.x;
										tlBulb[num_tlBulb][4] = scanStop.y + tolerance;
										tlBulb[num_tlBulb][5] = (tlBulb[num_tlBulb][4] - tlBulb[num_tlBulb][2]) * (tlBulb[num_tlBulb][3] - tlBulb[num_tlBulb][1]);
										tlBulb[num_tlBulb][6] = -1;
										tlBulb[num_tlBulb][7] = 2;
										tlBulb[num_tlBulb][8] = selectBody;
										inBulb[tlBody[8]] = num_tlBulb;
										tlBody[8]++;
										num_tlBulb++;
										anyMultiGreen = true;
									}
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
		for (int k = 0; k < tlBody[7]; k++)
		{
			scanStart.y = (locBox * k) + tlBody[2];
			checkStatus[k][0] = resBox + scanStart.y;
			checkStatus[k][1] = -1;
		}
		for (int l = 0; l < tlBody[8]; l++)
		{
			checkBox = round((float)(tlBulb[inBulb[l]][2] + tlBulb[inBulb[l]][4]) / 2);
			mindistBox = 9999;
			for (int k = 0; k < tlBody[7]; k++)
			{
				distBox = abs(checkBox - checkStatus[k][0]);
				if (distBox < mindistBox && checkStatus[k][1] == -1)
				{
					mindistBox = distBox;
					optimumBox = k;
				}
			}
			checkStatus[optimumBox][1] = l;
		}
		for (int k = 0; k < tlBody[7]; k++)
		{
			if (checkStatus[k][1] == -1)
			{
				tolerance = (int)round(((float)tlBody[3] - (float)tlBody[1]) / 3);
				if (tolerance < 1)
				{
					tolerance = 1;
				}
				scanStart.y = (locBox * k) + tlBody[2];
				scanStart.x = tlBody[1] + tolerance; 
				scanStop.y  = (locBox * (k + 1)) + tlBody[2];
				scanStop.x  = tlBody[3] - tolerance;
				for (int color = 0; color < 2; color++)
				{
					colorHist[color] = 0;
				}
				//Verify the class information.
				for (int j = scanStart.y; j < scanStop.y; j++)
				{
					for (int i = scanStart.x; i < scanStop.x; i++)
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
									//RED
									colorHist[0]++;
								}
								else if (aux_hue > 120 && aux_hue < 220){
									//GREEN
									colorHist[1]++;
								}
								else if (aux_hue > 320)
								{
									//RED
									colorHist[0]++;
								}
							}
						}
					}
				}

				maxHist = 0; bestColor = -1;
				for (int color = 0; color < 2; color++)
				{
					if (colorHist[color] > maxHist)
					{
						maxHist = colorHist[color];
						bestColor = color;
					}
				}

				if (bestColor > -1)
				{
					if (k < 2)
					{
						if (bestColor == 0 && anyMultiGreen == false)
						{
							if (anyRedYellow == false && maxHist > 5)
							{
								//add new bounding box.
								if (k == 0)
								{
									//cout << maxHist << " RED BB ADDED" << endl;
									tlBulb[num_tlBulb][0] = 2;
									tlBulb[num_tlBulb][1] = scanStart.x;
									tlBulb[num_tlBulb][2] = scanStart.y - tolerance;
									tlBulb[num_tlBulb][3] = scanStop.x;
									tlBulb[num_tlBulb][4] = scanStop.y + tolerance;
									tlBulb[num_tlBulb][5] = (tlBulb[num_tlBulb][4] - tlBulb[num_tlBulb][2]) * (tlBulb[num_tlBulb][3] - tlBulb[num_tlBulb][1]);
									tlBulb[num_tlBulb][6] = -1;
									tlBulb[num_tlBulb][7] = 0;
									tlBulb[num_tlBulb][8] = selectBody;
									inBulb[tlBody[8]] = num_tlBulb;
									tlBody[8]++;
									num_tlBulb++;
									anyRedYellow = true;
								}
								else
								{
									//cout << maxHist << " YELLOW BB ADDED" << endl;
									tlBulb[num_tlBulb][0] = 2;
									tlBulb[num_tlBulb][1] = scanStart.x;
									tlBulb[num_tlBulb][2] = scanStart.y - tolerance;
									tlBulb[num_tlBulb][3] = scanStop.x;
									tlBulb[num_tlBulb][4] = scanStop.y + tolerance;
									tlBulb[num_tlBulb][5] = (tlBulb[num_tlBulb][4] - tlBulb[num_tlBulb][2]) * (tlBulb[num_tlBulb][3] - tlBulb[num_tlBulb][1]);
									tlBulb[num_tlBulb][6] = -1;
									tlBulb[num_tlBulb][7] = 1;
									tlBulb[num_tlBulb][8] = selectBody;
									inBulb[tlBody[8]] = num_tlBulb;
									tlBody[8]++;
									num_tlBulb++;
									anyRedYellow = true;
								}
							}
						}
					}
					else
					{
						if (bestColor == 1)
						{
							if (anyMultiGreen == false && anyRedYellow == false)
							{
								if (anyGreen == true)
								{
									if (maxHist > 0)
									{
										//add new bounding box.
										//cout << maxHist << " GREEN BB ADDED" << endl;
										tlBulb[num_tlBulb][0] = 2;
										tlBulb[num_tlBulb][1] = scanStart.x;
										tlBulb[num_tlBulb][2] = scanStart.y - tolerance;
										tlBulb[num_tlBulb][3] = scanStop.x;
										tlBulb[num_tlBulb][4] = scanStop.y + tolerance;
										tlBulb[num_tlBulb][5] = (tlBulb[num_tlBulb][4] - tlBulb[num_tlBulb][2]) * (tlBulb[num_tlBulb][3] - tlBulb[num_tlBulb][1]);
										tlBulb[num_tlBulb][6] = -1;
										tlBulb[num_tlBulb][7] = 2;
										tlBulb[num_tlBulb][8] = selectBody;
										inBulb[tlBody[8]] = num_tlBulb;
										tlBody[8]++;
										num_tlBulb++;
										anyMultiGreen = true;
									}
								}
								else
								{
									if (maxHist > 10)
									{
										//add new bounding box.
										//cout << maxHist << " GREEN BB ADDED" << endl;
										tlBulb[num_tlBulb][0] = 2;
										tlBulb[num_tlBulb][1] = scanStart.x;
										tlBulb[num_tlBulb][2] = scanStart.y - tolerance;
										tlBulb[num_tlBulb][3] = scanStop.x;
										tlBulb[num_tlBulb][4] = scanStop.y + tolerance;
										tlBulb[num_tlBulb][5] = (tlBulb[num_tlBulb][4] - tlBulb[num_tlBulb][2]) * (tlBulb[num_tlBulb][3] - tlBulb[num_tlBulb][1]);
										tlBulb[num_tlBulb][6] = -1;
										tlBulb[num_tlBulb][7] = 2;
										tlBulb[num_tlBulb][8] = selectBody;
										inBulb[tlBody[8]] = num_tlBulb;
										tlBody[8]++;
										num_tlBulb++;
										anyMultiGreen = true;
									}
								}
							}
						}
					}
				}
			}	
		}
	}
}


void proposeTrafficLight(int tlBody[9], int selectBody, int inBulb[5], int &num_tlBulb, int tlBulb[90][9], Mat RGB)
{
	int scanPoint, scanRange[2], bbLimit[2];

	int lightPixel[5000][2], nPixel;
	//[0] = color spectrum
	//[1] = X or Y depending on selector
	
	selector = checkOrientation(tlBody, minDimension, maxDimension);
	adaptiveThreshold(tlBody, RGB, averageI, averageS);
	temp_var = minDimension / 2;
	if (selector == 1)
	{
		scanPoint = tlBody[2] + temp_var;
		bbLimit[0] = tlBody[2];
		bbLimit[1] = tlBody[4];
		scanRange[0] = tlBody[1];
		scanRange[1] = tlBody[3];
	}
	else
	{
		scanPoint = tlBody[1] + temp_var;
		bbLimit[0] = tlBody[1];
		bbLimit[1] = tlBody[3];
		scanRange[0] = tlBody[2];
		scanRange[1] = tlBody[4];
	}
			
	for (int color = 0; color < 2; color++)
	{
		colorHist[color] = 0;
	}

	nPixel = 0;
	int j = scanPoint;
	for (int i = scanRange[0]; i < scanRange[1]; i++)
	{
		//Identify the color of the pixel inside the bounding box.
		if (selector == 1)
		{
			B = (float)(RGB.at<Vec3b>(j, i)[0]) / 255;
			G = (float)(RGB.at<Vec3b>(j, i)[1]) / 255;
			R = (float)(RGB.at<Vec3b>(j, i)[2]) / 255;
		}
		else
		{
			B = (float)(RGB.at<Vec3b>(i, j)[0]) / 255;
			G = (float)(RGB.at<Vec3b>(i, j)[1]) / 255;
			R = (float)(RGB.at<Vec3b>(i, j)[2]) / 255;
		}
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
					lightPixel[nPixel][0] = 0;
					lightPixel[nPixel][1] = i;
					nPixel++;
				}
				else if (aux_hue > 120 && aux_hue < 220){
					//GREEN
					lightPixel[nPixel][0] = 1;
					lightPixel[nPixel][1] = i;
					nPixel++;
				}
				else if (aux_hue > 300)
				{
					//RED
					lightPixel[nPixel][0] = 0;
					lightPixel[nPixel][1] = i;
					nPixel++;
				}
			}
		}
	}
	
	int candidateClass;
	int candidateRed[5], nRed = 0, maxRed = 0;
	int candidateGreen[50][5], nGreen = 0;

	int prLimit[2]; prLimit[0] = 0; prLimit[1] = 0;
	int dPixel, posCount = 0;
	float avClassCount = 0;
	for (int k = 0; k < nPixel; k++)
	{
		if (posCount == 0)
		{
			avClassCount = lightPixel[k][0];
			prLimit[0] = lightPixel[k][1];
			posCount = 1;
		}
		else
		{
			dPixel = lightPixel[k][1] - lightPixel[k - 1][1];
			if (dPixel < dPixel_th)
			{
				avClassCount += lightPixel[k][0];
				prLimit[1] = lightPixel[k][1];
				posCount++;
			}
			else
			{
				if (posCount > bulb_th)
				{
					candidateClass = round(avClassCount / posCount);
					if (candidateClass == 0)
					{
						if (prLimit[1] - prLimit[0] > maxRed)
						{
							maxRed = prLimit[1] - prLimit[0];
							candidateRed[0] = 0;
							if (selector == 1)
							{
								candidateRed[1] = prLimit[0];
								candidateRed[2] = bbLimit[0];
								candidateRed[3] = prLimit[1];
								candidateRed[4] = bbLimit[1];
							}
							else
							{
								candidateRed[1] = bbLimit[0];
								candidateRed[2] = prLimit[0];
								candidateRed[3] = bbLimit[1];
								candidateRed[4] = prLimit[1];
							}
							nRed = 1;
						}
					}
					else
					{
						candidateGreen[nGreen][0] = 1;
						if (selector == 1)
						{
							candidateGreen[nGreen][1] = prLimit[0];
							candidateGreen[nGreen][2] = bbLimit[0];
							candidateGreen[nGreen][3] = prLimit[1];
							candidateGreen[nGreen][4] = bbLimit[1];
						}
						else
						{
							candidateGreen[nGreen][1] = bbLimit[0];
							candidateGreen[nGreen][2] = prLimit[0];
							candidateGreen[nGreen][3] = bbLimit[1];
							candidateGreen[nGreen][4] = prLimit[1];
						}
						nGreen++;
					}
				}
				avClassCount = lightPixel[k][0];
				prLimit[0] = lightPixel[k][1];
				posCount = 1;
			}
		}
	}
	if (posCount > bulb_th)
	{
		candidateClass = round(avClassCount / posCount);
		if (candidateClass == 0 && nRed == 0)
		{
			if (prLimit[1] - prLimit[0] > maxRed)
			{
				maxRed = prLimit[1] - prLimit[0];
				candidateRed[0] = 0;
				if (selector == 1)
				{
					candidateRed[1] = prLimit[0];
					candidateRed[2] = bbLimit[0];
					candidateRed[3] = prLimit[1];
					candidateRed[4] = bbLimit[1];
				}
				else
				{
					candidateRed[1] = bbLimit[0];
					candidateRed[2] = prLimit[0];
					candidateRed[3] = bbLimit[1];
					candidateRed[4] = prLimit[1];
				}
				nRed = 1;
			}
		}
		else
		{
			candidateGreen[nGreen][0] = 1;
			if (selector == 1)
			{
				candidateGreen[nGreen][1] = prLimit[0];
				candidateGreen[nGreen][2] = bbLimit[0];
				candidateGreen[nGreen][3] = prLimit[1];
				candidateGreen[nGreen][4] = bbLimit[1];
			}
			else
			{
				candidateGreen[nGreen][1] = bbLimit[0];
				candidateGreen[nGreen][2] = prLimit[0];
				candidateGreen[nGreen][3] = bbLimit[1];
				candidateGreen[nGreen][4] = prLimit[1];
			}
			nGreen++;
		}
	}

	bulbDimension = 0;
	if (nRed == 1)
	{
		bulbDimension += candidateRed[selector + 2] - candidateRed[selector];
	}
	for (int k = 0; k < nGreen; k++)
	{
		bulbDimension += candidateGreen[selector + 2] - candidateGreen[selector];
	}
	if (nRed + nGreen > 0)
	{
		tlBody[0] = 1;
		minDimension = round(float(bulbDimension) / (float)(nRed + nGreen));
		possibleBulb = int(round(float(maxDimension) / float(minDimension)));
		if (possibleBulb < 3)
		{
			possibleBulb = 3;
			normDimension = maxDimension / 3;
		}
		else if (possibleBulb > 5)
		{
			possibleBulb = 5;
			normDimension = maxDimension / 5;
		}
		else
		{
			normDimension = maxDimension / possibleBulb;
		}
		tlBody[7] = possibleBulb;
	}
	tlBody[8] = 0;

	//RED SPECTRUM
	if (nRed == 1)
	{
		tempIDX = ceil((((float)(candidateRed[selector + 2] - candidateRed[selector]) / 2) + candidateRed[selector] - tlBody[selector]) / normDimension);
		if (tempIDX < 2)
		{
			tlBulb[num_tlBulb][0] = 1;
			tlBulb[num_tlBulb][1] = candidateRed[1];
			tlBulb[num_tlBulb][2] = candidateRed[2];
			tlBulb[num_tlBulb][3] = candidateRed[3];
			tlBulb[num_tlBulb][4] = candidateRed[4];
			tlBulb[num_tlBulb][5] = (candidateRed[4] - candidateRed[2]) * (candidateRed[3] - candidateRed[1]);
			tlBulb[num_tlBulb][6] = -1;
			tlBulb[num_tlBulb][7] = 0;
			tlBulb[num_tlBulb][8] = selectBody;
			inBulb[tlBody[8]] = num_tlBulb;
			tlBody[8]++;
			num_tlBulb++;
		}
		else
		{
			tlBulb[num_tlBulb][0] = 1;
			tlBulb[num_tlBulb][1] = candidateRed[1];
			tlBulb[num_tlBulb][2] = candidateRed[2];
			tlBulb[num_tlBulb][3] = candidateRed[3];
			tlBulb[num_tlBulb][4] = candidateRed[4];
			tlBulb[num_tlBulb][5] = (candidateRed[4] - candidateRed[2]) * (candidateRed[3] - candidateRed[1]);
			tlBulb[num_tlBulb][6] = -1;
			tlBulb[num_tlBulb][7] = 1;
			tlBulb[num_tlBulb][8] = selectBody;
			inBulb[tlBody[8]] = num_tlBulb;
			tlBody[8]++;
			num_tlBulb++;
		}
	}
	//GREEN SPECTRUM
	for (int k = 0; k < nGreen; k++)
	{
		tlBulb[num_tlBulb][0] = 1;
		tlBulb[num_tlBulb][1] = candidateGreen[k][1];
		tlBulb[num_tlBulb][2] = candidateGreen[k][2];
		tlBulb[num_tlBulb][3] = candidateGreen[k][3];
		tlBulb[num_tlBulb][4] = candidateGreen[k][4];
		tlBulb[num_tlBulb][5] = (candidateGreen[k][4] - candidateGreen[k][2]) * (candidateGreen[k][3] - candidateGreen[k][1]);
		tlBulb[num_tlBulb][6] = -1;
		tlBulb[num_tlBulb][7] = 2;
		tlBulb[num_tlBulb][8] = selectBody;
		inBulb[tlBody[8]] = num_tlBulb;
		tlBody[8]++;
		num_tlBulb++;
	}		
}

void finalizeTrafficLight(int selectBody, int tlBody[30][9], int inBulb[5], int tlBulb[90][9], bool finStat[6], bool preStat[6])
{
	bool nowStat[6]; bool checkStatus;
	int detClass, verClass;

	for (int k = 0; k < 6; k++)
	{
		nowStat[k] = false;
	}

	if (selectBody > -1)
	{
		int countGreen = 0;
		for (int k = 0; k < tlBody[selectBody][8]; k++)
		{
			if (tlBulb[inBulb[k]][7] == 2)
			{
				countGreen++;
			}
		}
		for (int k = 0; k < tlBody[selectBody][8]; k++)
		{
			detClass = tlBulb[inBulb[k]][6];
			verClass = tlBulb[inBulb[k]][7];
			if (detClass > -1)
			{
				if (detClass == verClass)
				{
					nowStat[detClass] = true;
				}
				else
				{
					if (detClass < 2)
					{
						if (verClass == 2)
						{
							if (countGreen > 1)
							{
								checkStatus = false;
								for (int l = 3; l < 6; l++)
								{
									if (preStat[l] == true)
									{
										nowStat[l] = true;
										preStat[l] = false;
										checkStatus = true;
									}
								}
								if (checkStatus == false)
								{
									nowStat[verClass] = true;
								}
							}
							else
							{
								nowStat[verClass] = true;
							}
						}
						else
						{
							nowStat[verClass] = true;
						}
					}
					else
					{
						if (verClass == 2)
						{
							nowStat[detClass] = true;
						}
						else
						{
							if (verClass < 2)
							{
								nowStat[detClass] = true;
							}
							else
							{
								nowStat[verClass] = true;
							}
						}
					}

				}
			}
			else
			{
				if (verClass < 2)
				{
					nowStat[verClass] = true;
				}
				else
				{
					if (countGreen > 1)
					{
						checkStatus = false;
						for (int l = 3; l < 6; l++)
						{
							if (preStat[l] == true)
							{
								nowStat[l] = true;
								preStat[l] = false;
								checkStatus = true;
							}
						}
						if (checkStatus == false)
						{
							nowStat[verClass] = true;
						}
					}
					else
					{
						nowStat[verClass] = true;
					}
				}
			}
		}
		for (int k = 0; k < 6; k++)
		{
			finStat[k] = nowStat[k];
		}
		cnvCounter++;
	}
	else
	{
		if (cnvCounter > cnvFrame_th)
		{
			for (int k = 0; k < 6; k++)
			{
				finStat[k] = preStat[k];
			}
		}
		else
		{
			for(int k = 0; k < 6; k++)
			{
				finStat[k] = 0;
			}
		}
		cnvCounter = 0;
	}
	for (int k = 0; k < 6; k++)
	{
		preStat[k] = finStat[k];
	}

	

}