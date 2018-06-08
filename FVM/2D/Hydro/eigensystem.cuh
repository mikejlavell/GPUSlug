#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>


#ifndef defeig
#define defeig

/*--------------- Eigenvalues --------------------*/
__host__ __device__ void eigenvalues(double4 V,double lambda[], int dir);

/*--------------- Right Eigenvectors --------------------*/ 
__host__ __device__ void right_eigenvectors(double4 V, bool conservative, 
			                    double4 reig[], int dir);
/*--------------- Left Eigenvalues --------------------*/
__host__ __device__ void left_eigenvectors(double4 V,bool conservative, double4 leig[], int dir);
#endif
