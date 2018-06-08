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
    int stid = tidy*M + tidx;
    int stidx1 = stid+1;
    int stidy1 = (tidy+1)*M + tidx;

    double dtx = dt/gr_dx;
    double dty = dt/gr_dy;
    double4 temp = {0.0, 0.0, 0.0, 0.0};
   
    if( tidx>0 && tidx<(N-1) && tidy>0 && tidy<(M-1))
    {
    	temp.x = U[stid].x - dtx*(Flux[stidx1].x - Flux[stid].x) 
   			   - dty*(Fluy[stidy1].x - Fluy[stid].x);
   	temp.y = U[stid].y - dtx*(Flux[stidx1].y - Flux[stid].y)
 			   - dty*(Fluy[stidy1].y - Fluy[stid].y);
    	temp.z = U[stid].z - dtx*(Flux[stidx1].z - Flux[stid].z)
			   - dty*(Fluy[stidy1].z - Fluy[stid].z);
    	temp.w = U[stid].w - dtx*(Flux[stidx1].w - Flux[stid].w)
			   - dty*(Fluy[stidy1].w - Fluy[stid].w);
    }
    U[stid] = temp;
    
    cons2prim(U[stid], V[stid]);
	//}
    //}
}


