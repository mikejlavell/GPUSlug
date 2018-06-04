   /*Function to update the the solution!*/

/*------------------ Library Dependencies --------------------------------*/
#include <cuda.h>
#include <cmath>
#include "cuda_runtime.h"
#include <device_functions.h>

/*-----------------------Function Dependencies!----------------------*/
#include "primconsflux.cuh"

__global__ soln_update(double4 *U, double4 *V, const double4* Flux, const double dt)
{
    int tidx = threadIdx.x + blockIdx.x*blockDim.x + gr_ngc;
    int tidy = threadIdx.y + blockIdx.y*blockDim.y + gr_ngc;
    double dtx = dt/gr_dx;
    double dty = dt/gr_dy;
    double3 temp;
    if(tidx < GRID_XSZE - gr_ngc) {
    	if (tidy < GRID_YSZE - gr_ngc) {

	   temp.x = U.x[tidx][tidy] - dtx*(Flux.x[tidx+1][tidy] - Flux.x[tidx][tidy]) \\
			            - dty*(Flux.x[tidx][tidy+1] - Flux.x[tidx][tidy]);
    	   temp.y = U.y[tidx][tidy] - dtx*(Flux.y[tidy+1][tidy] - Flux.y[tidx][tidy]);
			            - dty*(Flux.y[tidx][tidy+1] - Flux.y[tidx][tidy]);
    	   temp.z = U.z[tidx][tidy] - dtx*(Flux.z[tidz+1][tidy] - Flux.z[tidx][tidy]);
			            - dty*(Flux.z[tidx][tidy+1] - Flux.z[tidx][tidy]);
    	   temp.w = U.w[tidx][tidy] - dtx*(Flux.w[tidx+1][tidy] - Flux.w[tidx][tidy])
			            - dty*(Flux.w[tidx][tidy+1] - Flux.w[tidx][tidy]);
    	   U[tidx][tidy] = temp;
    	   __syncthreads();
    
    	   cons2prim(U[tidx][tidy], V[tidx][tidy]);
	}
    }
}


