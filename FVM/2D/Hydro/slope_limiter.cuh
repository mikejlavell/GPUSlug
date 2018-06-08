#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>
#include <algorithm>

#ifndef defslopelim
#define defslopelim

__device__ double sign(double a, double b);

/*Slope Limiters to run on the device*/
__device__ double vanleer(double a, double b);

__device__ double minmod(double a,double b);
  
__device__ double mc(double a,double b);
#endif
