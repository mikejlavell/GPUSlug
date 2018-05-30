/* This header file contains the functions calculate the Equation of State*/

#include <cuda_runtime.h>
#include <stdio.h>

#include "definition.h"


__device__ eos_cell(const double U,const double eint, double &pres)
{
       pres = max((sim_gamma-1.)*dens*eint,sim_smallPres);
}
