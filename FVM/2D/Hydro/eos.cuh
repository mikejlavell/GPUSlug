/* This header file contains the functions calculate the Equation of State*/

#include <cuda_runtime.h>
#include <stdio.h>
#include "definition.h"

#ifndef defeos
#define defeos

__host__ __device__ void eos_cell(const double dens, const double eint, double &pres);

#endif
