   /*Function to update the the solution!*/

/*------------------ Library Dependencies --------------------------------*/
#include <cuda.h>
#include <cmath>
#include "cuda_runtime.h"
#include <device_functions.h>

/*-----------------------Function Dependencies!----------------------*/
#include "primconsflux.cuh"
#include "definition.h"

__global__ void soln_update(double4 *U, double4 *V, const double4 *Flux,
			    const double4 *Fluy, const double dt)
{
    int tidx = threadIdx.x + blockIdx.x*blockDim.x + gr_ngc;
    int tidy = threadIdx.y + blockIdx.y*blockDim.y + gr_ngc;
    double dtx = dt/gr_dx;
    double dty = dt/gr_dy;
    double4 temp = {0.0, 0.0, 0.0, 0.0};
    //if(tidx < GRID_XSZE - gr_ngc) {
    //	if (tidy < GRID_YSZE - gr_ngc) {

    temp.x = U[tidx][tidy].x - dtx*(Flux[tidx+1][tidy].x - Flux[tidx][tidy].x) 
   		             - dty*(Fluy[tidx][tidy+1].x - Fluy[tidx][tidy].x);
    temp.y = U[tidx][tidy].y - dtx*(Flux[tidx+1][tidy].y - Flux[tidx][tidy].y)
 		             - dty*(Fluy[tidx][tidy+1].y - Fluy[tidx][tidy].y);
    temp.z = U[tidx][tidy].z - dtx*(Flux[tidx+1][tidy].z - Flux[tidx][tidy].z)
			     - dty*(Fluy[tidx][tidy+1].z - Fluy[tidx][tidy].z);
    temp.w = U[tidx][tidy].w - dtx*(Flux[tidx+1][tidy].w - Flux[tidx][tidy].w)
			     - dty*(Fluy[tidx][tidy+1].w - Fluy[tidx][tidy].w);
    syncthreads();

    U[tidx][tidy] = temp;
    __syncthreads();
    
    cons2prim(U[tidx][tidy], V[tidx][tidy]);
	//}
    //}
}


