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
double cfl_omp(double4 *gr_V)
{   
    double cs, lambda, minDt, dt;
    minDt = 1.e20; 
    
#pragma omp parallel for reduction(max:maxSpeed) num_threads(ncores) private(cs, lambda)
 for(int i = gr_ibeg, gr_iend)
    {
	 for(int j = gr_jbeg, gr_jend)
	 {            
         	cs = sqrtf(sim_gamma*gr_V.w[i][j]/gr_V.x[i][j]);
         	lambdax=abs(gr_V.y[i][j]) + cs;
		lambday=abs(gr_V.z[i][j]) + cs;
         	minDt = min(minDt, gr_dx/lambdax, gr_dy/lambday):
	 }
     }

  // cfl
  dt = cfl*minDt;
  return dt;
}




