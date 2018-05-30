/*gpuSlugCode/cfl.cu

Routine discovers dt. CUDA reduce algorithm is
used to efficiently discover stable step sizes.

Written by Michael Lavell, University of California, Santa Cruz
May 2018

remarks

- how to I access values of double4 (.u,.x,..)

*/

/*------------- Library Dependencies ------------------*/
#include <cuda.h>
#include <cmath.h>
#include "cuda_runttime.h"
#include <device_functions.h>


/*------------- Code Dependencies ---------------------*/
#include "definition.h"
#include "primconsflux.cuh"

/*------------------ CFL  Time Step- ------------------*/
 double cfl_cuda(double4 *gr_V)
{

	size_t sz = N*sizeof(double4);
	size_t sz_reduc = N/BLOCK_DIMX*sizeof(double4);
	float *d_V, *d_dt_reduc;
	float *d_dt;

	// Allocate memory
	cudaMalloc((void**)&d_V, sz);
	cudaMalloc((void**)&d_dt_reduc, sz_reduc);

	// Kernel parameters
	dim3 block(BLOCK_DIMX,BLOCK_DIMY,1);
	dim3 grid(ceil(GRID_XSZE/block.x), ceil(GRID_YSZE/block.y , 1);	
	
	size_t sz_shar = block.x*sizeof(double);

	// Transfer data
	cudaMemcpy(d_V, gr_V, sz, cudaMemcpyHostToDevice);

	// Find array of reduced dt
	cfl_reduce<<<grid,block,sz_shar>>>(d_V,d_dt_reduc);
	cudaDeviceSyncrhonize();
	
	// Call CFL kernel to get final dt
	dt_reduce<<<1,block>>>(d_dt_reduc,d_dt)
	cudaDeviceSynchronize();

	// Transfer data
	cudaMemcpy(dt, d_dt, sizeof(double), cudaMemcpyDeviceToHost);

	// Deallocate memory
	cudaFree(d_V);
	cudaFree(d_dt_reduc);

	return dt;

} 


/*-----------------  CFL Reduce ------------------------*/
__global__ void cfl_reduce(double4 *V, double dt_reduc){

	//Shared data is allocated in kernel call via dynamic shared memory
	extern __shared__ double shar_dt[];
	
	// Initialize minDt
	double minDt = 1e30;
	double cs, lambda;
			
	// Block coordinates
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	// Calculate dt
	cs = sqrt(sim_gamma * V.z[tid]/V.u[tid]);
	u = abs(V.x) + cs;
	v = abs(V.y) + cs;
			
	shar_dt[tid] = min(minDt, min(gr_dx/u, gr_dy/v));

	// Find min on shared memory
	for( int i = blockDim.x/2; s>0; s>>=1 )
	{
		if( tid<s )
		{
			if( minDt > shar_dt[tid])
				minDt = shar_dt[tid];
		}
		__syncthreads();
	}

	// One thread writes result
	if ( tid==0 )
		dt_reduc = minDt;

}


/*-----------------  dt Reduce ------------------------*/
__global__ void dt_reduce(double *dt_reduc, double dt){
	
	extern __shared__ double shar_dt[];

	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	double minDt = 2e10;

	for int i = blockDim.x/2; s>0; s>>=1 )
	{
		if( tid<s )
		{
			if (minDt > shar_dt[tid])
				minDt = shar_dt[tid];
		}			
		__syncthreads();
	}
	
	if ( tid==0 )
		dt = minDt;
	
}



/*----------------- CFL Condition ---------------------*/
double cfl_omp(double3 *gr_V)
{   
    double cs, lambda, maxSpeed, dt; 
    maxSpeed = -1e30;
#pragma omp parallel for reduction(max:maxSpeed) num_threads(ncores) private(cs, lambda)
 for(int i = gr_ibeg, gr_iend)
    {
            
         cs = sqrtf(sim_gamma*gr_V.z[i]/gr_V.x[i]);
         lambda=(abs(gr_V.y[i]) + cs);
         maxSpeed=max(maxSpeed,lambda);
     }

  // cfl
  dt = 0.8*gr_dx/maxSpeed;
  return dt;
}




