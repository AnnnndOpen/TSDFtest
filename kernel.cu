
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <stdio.h>
#include <device_functions.h>

//#include "device_launch_parameters.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cmath>
#include <time.h>
using namespace std;

const int range = 128;
const int workPerThread = 64;//cannot be smaller than the range
int halfRange = range / 2;
float cube[range*range*range];
int weight[range*range*range];

bool ou[range][range][range];
//float p = 0.8;

__global__
void calculate(float *cubecu, int *weightcu, float *pixelcu, float *rotcu,float mx,float my,float mz,int range,int workPerThread)// float *mxcu, float *mycu, float *mzcu, int *rangecu)
{
	float pos1[3];
	float pos2[3];
	int pos3[3];
	int i, j, k,l,m;
	float tmp1;
	int posInCube;
	int zdim;
	__shared__ float rotcus[9];//,range;
	__shared__ float pixelcus[10000];
	int xtot = 99;
	int ytot = 99;
	int total = xtot*ytot;
	int pixelstart= (threadIdx.y*blockDim.x + threadIdx.x)*64;
	int pixelend=pixelstart+64;
	if (pixelend > total)
		pixelend = total;
	for (i=pixelstart;i<pixelend;i++)//parallel fetch the depth image
		pixelcus[i] = pixelcu[i];
	if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
	{
		for (i = 0; i < 9; i++)
			rotcus[i] = rotcu[i];
		//for (i = 0; i <total; i++)
			//pixelcus[i] = pixelcu[i];
	}
	__syncthreads();
	i = blockDim.x*blockIdx.x + threadIdx.x;
	j = blockDim.y*blockIdx.y + threadIdx.y;
	for (zdim = 0; zdim < workPerThread; zdim++)
	{
		//k = blockDim.z*blockIdx.z + threadIdx.z;
		k = workPerThread * blockIdx.z + zdim;
		//cubecu[k*range*range + j*range + i] = rotcus[0];
		pos1[0] = i*0.2 / (range)-0.1 - mx;
		pos1[1] = j*0.2 / (range)-my;
		pos1[2] = k*0.2 / (range)-0.1 - mz;
		for (l = 0; l < 3; l++)
		{
			pos2[l] = 0;
			for (m = 0; m < 3; m++)
			{
				pos2[l] = pos2[l] + rotcus[l * 3 + m] * pos1[m];
			}
		}
		pos3[0] = int((pos2[0] + 0.1) / 0.002 + 0.5);
		pos3[1] = int(pos2[1] / 0.002 + 0.5);
		//if ((blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) && (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
			//cubecu[k*range*range + j*range + i]=rotcus[0];
			//cubecu[0] = 2.0;
		if (((pos3[0] * ytot + pos3[1]) < xtot*ytot) && ((pos3[0] * ytot + pos3[1]) >= 0))
			if (pixelcus[pos3[0] * ytot + pos3[1]] > -0.01)
			{
				//cubecu[k*range*range + j*range + i] = range;
				tmp1 = (pixelcus[pos3[0] * ytot + pos3[1]] - pos2[2]) / 0.001;
				if ((tmp1 > 1) || (tmp1 < -1))
					tmp1 = 0;
				posInCube = i*range*range + j*range + k;
				cubecu[posInCube] = (cubecu[posInCube] * weightcu[posInCube] + tmp1) / (weightcu[posInCube] + 1);
				weightcu[posInCube] = weightcu[posInCube] + 1;
			}
	}
}

void pro(string a1, float *cubecu, int *weightcu)//calculate rotation matrix and call the 'calculate' function
{
	int xtot, ytot;
	int i, j;
	float mx, my, mz, rx, ry, rz,an;
	float rot[9];
	float pixel[100 * 100];
	float *pixelcu, *rotcu;
	a1 += "data.txt";
	ifstream fin1(a1);
	fin1 >> mx;
	fin1 >> my;
	fin1 >> mz;
	fin1 >> rx;
	fin1 >> ry;
	fin1 >> rz;
	fin1 >> an;
	fin1 >> xtot;
	fin1 >> ytot;
	for (i = 0; i<xtot; i++)
		for (j = 0; j<ytot; j++)
		{
			fin1 >> pixel[j+i*ytot];
		}
	fin1.close();

	rot[0] = 1 - 2 * ry*ry - 2 * rz*rz;
	rot[1] = 2 * rx*ry - 2 * rz*an;
	rot[2] = 2 * rx*rz + 2 * ry*an;
	rot[3] = 2 * rx*ry + 2 * rz*an;
	rot[4] = 1 - 2 * rx*rx - 2 * rz*rz;
	rot[5] = 2 * ry*rz - 2 * rx*an;
	rot[6] = 2 * rx*rz - 2 * ry*an;
	rot[7] = 2 * ry*rz + 2 * rx*an;
	rot[8] = 1 - 2 * rx*rx - 2 * ry*ry;
	dim3 gb(range/32, range/32, range/workPerThread);
	dim3 tb(32, 32, 1);//I set the dimension of x and y as 32 because the max threadNum per block is 1024
	cudaMalloc((void **)&pixelcu, xtot*ytot * sizeof(float));
	cudaMemcpy(pixelcu, pixel, xtot*ytot*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&rotcu, 9 * sizeof(float));
	cudaMemcpy(rotcu, rot, 9 * sizeof(float), cudaMemcpyHostToDevice);

	calculate <<<gb, tb >>> (cubecu, weightcu, pixelcu, rotcu, mx, my, mz, range,workPerThread);
	//cudaFree(&mxcu); cudaFree(&mycu); cudaFree(&mzcu); cudaFree(&rangecu);
	cudaThreadSynchronize();
	cudaFree(pixelcu);
	cudaFree(rotcu);
}

int main()
{
	char tmp;
	int totalVertexOut;
	int i, j, k;
	float *cubecu;
	int *weightcu;

	clock_t startt = clock();
	if (cudaSuccess != cudaMalloc((void **)&cubecu, range*range*range * sizeof(float)))
		cout << "error1";
	if (cudaSuccess != cudaMemset(cubecu, 0, range*range*range * sizeof(float)))
		cout << "error2";
	
	if (cudaSuccess != cudaMalloc((void **)&weightcu, range*range*range * sizeof(int)))
		cout << "error3";

	if (cudaSuccess != cudaMemset(weightcu, 0, range*range*range * sizeof(int)))
		cout << "error4";
	//----------------load the pixels and calculate the 3D matrix-----------------------

	pro("bun000",cubecu,weightcu);
	pro("bun090",cubecu,weightcu);
	pro("bun180", cubecu, weightcu);
	pro("bun270", cubecu, weightcu);
	pro("bun045", cubecu, weightcu);
	pro("ear_back", cubecu, weightcu);
	pro("top2", cubecu, weightcu);
	pro("top3", cubecu, weightcu);
	pro("bun315", cubecu, weightcu);
	pro("chin", cubecu, weightcu);
	//-----------------------------------------------------
	cudaMemcpy(cube, cubecu, range*range*range * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(weight, weightcu, range*range*range * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(cubecu);
	cudaFree(weightcu);
	tmp = char(10);
	ofstream fout1("bunoutparallel.ply");
	totalVertexOut = 0;

	//----------------struct the final output model------------------------

	for (i = 0; i<range - 1; i++)
		for (j = 0; j<range - 1; j++)
			for (k = 0; k<range - 1; k++)
			{
				ou[i][j][k] = false;
				if (weight[i*range*range+j*range+k]>0)
				{
					if (weight[(i+1)*range*range+j*range+k]>0)
						if (cube[i*range*range + j*range + k] * cube[(i + 1)*range*range + j*range + k]<0)
							ou[i][j][k] = true;
					if (weight[i*range*range+(j+1)*range+k]>0)
						if (cube[i *range*range + j*range + k] * cube[i*range*range + (j + 1)*range + k]<0)
							ou[i][j][k] = true;
					if (weight[i*range*range +j*range+(k+1)]>0)
						if (cube[i*range*range + j*range + k] * cube[i*range*range + j*range + (k + 1)]<0)
							ou[i][j][k] = true;
					if (ou[i][j][k] == true)
						totalVertexOut++;
				}
				else
					ou[i][j][k] = false;
			}

	//-----------------------------------------------------------
	clock_t endt = clock();
	fout1 << "ply" << tmp << "format ascii 1.0" << tmp;
	fout1 << "element vertex " << totalVertexOut << tmp;
	fout1 << "property float x" << tmp;
	fout1 << "property float y" << tmp;
	fout1 << "property float z" << tmp;
	fout1 << "end_header" << tmp;

	for (i = 0; i<range - 1; i++)
		for (j = 0; j<range - 1; j++)
			for (k = 0; k<range - 1; k++)
				if (ou[i][j][k] == true)
				{
					fout1 << i - halfRange << ' ' << j - halfRange << ' ' << k - halfRange << tmp;
				}

	fout1.close();
	ofstream fout2("calculateTime");
	double time = (double)(endt - startt) / CLOCKS_PER_SEC;
	fout2 << "cubesize:" << range << '*' << range << '*' << range << '\n';
	fout2 << "time used (without initialize and output part):" << time << '\n';
	fout2.close();
	return 0;
}
