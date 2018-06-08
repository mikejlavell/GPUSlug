/*gpuSlugCode/cfl.cu

Routine discovers dt. CUDA reduce algorithm is
used to efficiently discover stable step sizes.

Written by Michael Lavell, University of California, Santa Cruz
May 2018

*/

#ifndef defcfl
#define defcfl
/*------------- Code Dependencies ---------------------*/
#include "definition.h"

/*------------------ CFL  Time Step- ------------------*/
 double cfl_cuda(double4 *gr_V); 


/*-----------------  CFL Reduce ------------------------*/
__global__ void cfl_reduce(double4 *V, double *dt_reduc);

/*-----------------  dt Reduce ------------------------*/
__global__ void dt_reduce(double *dt_reduc, double *dt);
/*----------------- CFL Condition ---------------------*/
double cfl_omp(double4 *gr_V);

#endif

