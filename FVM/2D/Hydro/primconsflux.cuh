/* This header file contains the functions to Switch Variable Type and Calculate Flux*/


#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>
#include <algorithm>

#ifndef defprimcons
#define defprimcons


//GPU/host function to calculate the conservative variables given the primitive vars
__host__ __device__  void prim2cons(const double4 V, double4 &U);


//GPU/host function to calculate the primitive variables given the conservative vars
__device__  void cons2prim(const double4 U,double4 &V);
   

//GPU/host function to calculate the analytic flux from a primitive variables
__device__  void prim2flux(const double4 V,double4 &Flux, const int dir);

    
//GPU/host function to calculate the analytic flux from a conservative variables
__device__  void cons2flux(const double4 U, double4 &Flux, int dir);

#endif


