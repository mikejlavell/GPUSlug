/*gpuSlugCode/cfl.cu

Routine discovers dt. CUDA reduce algorithm is
used to efficiently discover stable step sizes.

Written by Michael Lavell, University of California, Santa Cruz
May 2018

*/
/*------------- Code Dependencies ---------------------*/
#include "definition.h"
#include "cfl.cuh"
/*------------------ CFL  Time Step- ------------------*/
 double cfl_cuda(double4 *d_V)
{

	size_t sz = GRID_XSZE*GRID_YSZE*sizeof(double4);
	size_t sz_reduc = (GRID_XSZE/BLOCK_DIMX)*(GRID_YSZE/BLOCK_DIMY)*sizeof(double4);
	double *d_dt_reduc, *d_dt,*h_dt;
	double dt=0.0;

	// Allocate memory
	cudaMalloc((void**)&d_dt,sizeof(double));
	cudaMalloc((void**)&d_dt_reduc, sz_reduc);
	h_dt = (double*)malloc(sizeof(double));

	// Kernel parameters
	dim3 block(BLOCK_DIMX*BLOCK_DIMY,1,1);
	dim3 grid(ceil(GRID_XSZE/block.x)* ceil(GRID_YSZE/block.y),1 , 1);	
	
	size_t sz_shar = block.x*sizeof(double);

	// Find array of reduced dt
	cfl_reduce<<<grid,block,sz_shar>>>(d_V,d_dt_reduc);
	cudaDeviceSynchronize();
	
	// Call CFL kernel to get final dt
	dt_reduce<<<1,block,sz_shar>>>(d_dt_reduc,d_dt);
	cudaDeviceSynchronize();

	// Transfer data
	cudaMemcpy(h_dt, d_dt, sizeof(double), cudaMemcpyDeviceToHost);
	dt = h_dt[0];

	// Deallocate memory
	cudaFree(d_dt_reduc);
	cudaFree(d_dt);
	free(h_dt);

	return dt;

} 


/*-----------------  CFL Reduce ------------------------*/
__global__ void cfl_reduce(double4 *V, double *dt_reduc){

	//Shared data is allocated in kernel call via dynamic shared memory
	extern __shared__ double shar_dt[];
	
	// Initialize minDt
	double minDt = 1e30;
	double cs,u,v;
			
	// Block coordinates
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	// Calculate dt
	cs = sqrt(sim_gamma * V[myId].w/V[myId].x);
	u = abs(V[myId].y) + cs;
	v = abs(V[myId].z) + cs;
			
	shar_dt[tid] = min(minDt, min(gr_dx/u, gr_dy/v));
	__syncthreads();

	// Find min on shared memory
	for( int s = blockDim.x/2; s>0; s>>=1 )
	{
		if( tid<s )
		{
			shar_dt[tid] = min(shar_dt[tid],shar_dt[tid+s]);
		}
		__syncthreads();
	}

	// One thread writes result
	if ( tid==0 )
		dt_reduc[blockIdx.x] = shar_dt[0];

}


/*-----------------  dt Reduce ------------------------*/
__global__ void dt_reduce(double *dt_reduc, double *dt){
	
	extern __shared__ double shar_dt[];

	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	double minDt = 2e10;

	shar_dt[tid] = min(minDt,dt_reduc[myId]);

	for( int s = blockDim.x/2; s>0; s>>=1 )
	{
		if( tid<s )
		{
			shar_dt[tid] = min(shar_dt[tid],shar_dt[tid+s]);
		}			
		__syncthreads();
	}
	
	if ( tid==0 )	dt[0] = shar_dt[0];
	
}



/*----------------- CFL Condition ---------------------*/
double cfl_omp(double4 *gr_V)
{   
    double cs, lambdax, lambday, minDt, dt;
    minDt = 1.e20; 
    
#pragma omp parallel for reduction(max:maxSpeed) num_threads(ncores) private(cs, lambda)
 for(int i = gr_ibeg; i < gr_iend; i++)
    {
	 for(int j = gr_jbeg; j < gr_jend; j++)
	 {            
		int stid = j*M+i;

         	cs = sqrtf(sim_gamma*gr_V[stid].w/gr_V[stid].x);
         	lambdax=abs(gr_V[stid].y) + cs;
		lambday=abs(gr_V[stid].z) + cs;
         	minDt = min(minDt, min(gr_dx/lambdax, gr_dy/lambday));
	 }
     }

  // cfl
  dt = sim_cfl*minDt;
  return dt;
}



