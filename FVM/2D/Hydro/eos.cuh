/* This header file contains the functions calculate the Equation of State*/

#include <cuda_runtime.h>
#include <stdio.h>



__host__ __device__ void eos_cell(const double dens, const double eint, double &pres, double gamma)
{
       pres = fmax((gamma-1.)*dens*eint,1e-6);
}
