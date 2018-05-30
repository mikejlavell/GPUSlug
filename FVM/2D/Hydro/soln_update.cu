   /*Function to update the the solution!*/

/*------------------ Library Dependencies --------------------------------*/
#include <cuda.h>
#include <cmath>
#include "cuda_runtime.h"
#include <device_functions.h>

/*-----------------------Function Dependencies!----------------------*/
#include "primconsflux.cuh"

__global__ soln_update(double3 *U,double3 *V, const double3* Flux,const double dt)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x + gr_ngc;
    double3 temp;
    if(tid < GRID_SIZE - gr_ngc) {
    
    temp.x = U.x[tid] - dtx*(Flux.x[i+1] - Flux.x[tid]);
    temp.y = U.y[tid] - dtx*(Flux.y[i+1] - Flux.y[tid]);
    temp.z = U.z[tid] - dtx*(Flux.z[i+1] - Flux.z[tid]);
    
    U[tid] = temp;
    __syncthreads();
    
    cons2prim(U[tid], V[tid]);
    }
}


