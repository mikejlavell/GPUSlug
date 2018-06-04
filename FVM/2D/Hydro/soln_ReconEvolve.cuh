/*------------------------------- Library Dependencies ---------------------------*/
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>
#include <algorithm>
 
/*------------------------------ Program Dependencies ----------------------------*/
#include "definition.h"
#include "slope_limiter.cuh"
 
__device__ double dot_product(double4 u, double4 v){
    double ans = 0.0;
    ans = u.x*v.x + u.y*v.y + u.z*v.z u.w*v.w;
    return ans;
 }
 
/*------------------------------ Reconstruction Kernel ----------------------------*/
 __global__ void soln_reconstruct_PLM(const double dt, double4 *V, double4 *vl,
  double4 *vr, int dir)
 {
    if( dir == 0 ){
	//int GRID_SIZE = GRID_XSZE;
	double gr_dX = gr_dx;
	int a = 1; int b = 0;
	}
    else if( dir == 1){
	//int GRID_SIZE = GRID_YSZE;
	double gr_dX = gr_dy;
	int a = 0; int b = 1;
	}

    int tidx = threadIdx.x + blockIdx.x*blockDim.x + gr_ngc;
    int tidy = threadIdx.y + blockIdx.y*blockDim.y + gr_ngc;
    if(tidx < GRID_XSZE - gr_ngc){
    if(tidy < GRID_YSZE - gr_ngc){
    double lambdaDtDx;
    double lambda[NUMB_WAVE] = {0.0};
    double4 leig[NUMB_WAVE] = {0.0};
    double4 reig[NUMB_WAVE] = {0.0};
    double4 delL = {0.0}, delR= {0.0}; 
    double pL = 0.0, pR = 0.0, delW[NUMB_WAVE] = {0.0};

    // Calculate eigensystem
    eigenvalues(V[tidx][tidy],lambda); //Get Eigenvalues for reconstruction
    left_eigenvectors (V[tidx][tidy],0,leig);
    right_eigenvectors(V[tidx][tidy],0,reig);
    __syncthreads();

        for(int kWaveNum = 0; kWaveNum < NUMB_WAVE; kWaveNum ++)
        {
           delL.x = V[tidx][tidy].x-V[tidx-a][tidy-b].x;
           delL.y = V[tidx][tidy].y-V[tidx-a][tidy-b].y;
           delL.z = V[tidx][tidy].z-V[tidx-a][tidy-b].z;
           delL.w = V[tidx][tidy].w-V[tidx-a][tidy-b].w;
           delR.x = V[tidx+a][tidy+b].x-V[tidx][tidy].x;
           delR.y = V[tidx+a][tidy+b].y-V[tidx][tidy].y;
           delR.z = V[tidx+a][tidy+b].z-V[tidx][tidy].z;
           delR.w = V[tidx+a][tidy+b].w-V[tidx][tidy].w;
	   __syncthreads();

           // project onto characteristic vars
           pL = dot_product(leig[kWaveNum], delL);
           pR = dot_product(leig[kWaveNum], delR);

           // Use a TVD Slope limiter
           if (sim_limiter == 1){
                del[kWaveNum] = minmod(pL, pR):
              }
           else if (sim_limiter == 2){
                del[kWaveNum] = vanLeer(pL, pR);
              }
           else if (sim_limiter == 3){
                del[kWaveNum = mc(pL, pR);
              }
	   __syncthreads();
        }   
        
       //do char tracing
       //set the initial sum to be zero
       double4 sigL= {0.0,0.0,0.0,0.0};
       double4 sigR= {0.0,0.0,0.0,0.0};

        if (sim_riemann == "roe"){
	   for(int kWaveNum = 0; kWaveNum<NUMB_WAVE; kWaveNum++){
	      lambdaDtDx = lambda[kWaveNum]*dt/gr_dX;
              if (lambda(kWaveNum) > 0){
              //Right Sum
                 sigR.x += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].x*delW[kWaveNum];
        	 sigR.y += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].y*delW[kWaveNum];
		 sigR.z += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].z*delW[kWaveNum];
		 sigR.w += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].w*delW[kWaveNum];
		 __syncthreads();        
	      }
              else if (lambda(kWaveNum) < 0){
              //Left Sum
                 sigL.x += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].x*delW[kWaveNum];
                 sigL.y += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].y*delW[kWaveNum];
                 sigL.z += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].z*delW[kWaveNum];
                 sigL.w += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].w*delW[kWaveNum];
              	 __syncthreads();
	      }
	    }
         }
         else if (sim_riemann == "hll" || sim_riemann == "hllc")
	 {
	     for(int kWaveNum = 0; kWaveNum<NUMB_WAVE; kWaveNum++)
             {
		lambdaDtDx = lambda[kWaveNum]*dt/gr_dX;
             	//Right Sum
             	sigR.x += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].x*delW[kWaveNum];
	     	sigR.y += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].y*delW[kWaveNum];
	     	sigR.z += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].z*delW[kWaveNum];
	     	sigR.w += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].w*delW[kWaveNum];
		//Left Sum
             	sigL.x += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].x*delW[kWaveNum];
             	sigL.y += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].y*delW[kWaveNum];
             	sigL.z += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].z*delW[kWaveNum];
             	sigL.w += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].w*delW[kWaveNum];
	    	__syncthreads();
	      }     	
	 }

         // Now PLM reconstruction for dens, velx, vely, and pres
         vl[tidx][tidy].x = V[tidx][tidy].x + sigL.x;
         vl[tidx][tidy].y = V[tidx][tidy].y + sigL.y;
         vl[tidx][tidy].z = V[tidx][tidy].z + sigL.z;
         vl[tidx][tidy].w = V[tidx][tidy].w + sigL.w;
         vr[tidx][tidy].x = V[tidx][tidy].x + sigR.x;
         vr[tidx][tidy].y = V[tidx][tidy].y + sigR.y;
         vr[tidx][tidy].z = V[tidx][tidy].z + sigR.z;   
         vr[tidx][tidy].w = V[tidx][tidy].w + sigR.w;   
	 }
     }
 }}

/*---------------------------------- Get Flux Kernel ---------------------------*/ 
__global__ void soln_getFlux(double dt, double4 *vl, double4 *vr, double4 *flux, int dir)
{
	if( dir == 0) a = 1; b = 0;
	else if( dir == 1) a = 0; b = 1;

	int tidx = threadIdx.x + blockIdx.x*blockDim.x + gr_ngc;
	int tidy = threadIdx.y + blockIdx.y*blockDim.y + gr_ngc;
	if(tidx < GRID_XSZE - gr_ngc){
	if(tidy < GRID_YSZE - gr_ngc) 
	{
/*		if(sim_riemann == "hll")
		{
			//Call the HLL Riemann Solver
			hll(vr[tidx-a][tidy-b],vl[tidx][tidy],flux[tidx][tidy]); 
		}
		if(sim_riemann == "hllc")
		{
			//Call the HLLC Riemann Solver
			hllc(vr[tidx-a][tidy-b],vl[tidx][tidy],flux[tidx][tidy]);
		}//*/
		if(sim_riemann == "roe")
		{
			//Call the Roe Riemann Solver
			roe(vr[tidx-a][tidy-b],vl[tidx][tidy],flux[tidx][tidy]); 
		}
	}}
}

/*-------------------  Solution Reconstruct Evolve and Average -----------------*/
void soln_ReconEvolveAvg(double dt, double4 *d_gr_V, double4 *d_gr_U,
 double4 * d_gr_vlx, double4 *d_gr_vrx, double4 *d_gr_fluxx,
 double4 * d_gr_vly, double4 *d_gr_vry, double4 *d_gr_fluxy, dim3 grid, dim3 block) 
 {
    // Reconstruct in x-direction
    soln_reconstruct_PLM<<<grid,block>>>(dt, d_gr_V, d_gr_vlx, d_gr_vrx, 0);
    CudaCheckError();
    cudaDeviceSynchronize();

    // Get flux in x-direction
    soln_getFlux<<<grid,block>>>(dt, d_gr_vly, d_gr_vry, d_gr_fluxx, 0)
    CudaCheckError();
    cudaDeviceSynchronize();

    // Reconstruct in y-direction
    soln_reconstruct_PLM<<<grid,block>>>(dt, d_gr_V, d_gr_vly, d_gr_vry, 1);
    CudaCheckError();
    cudaDeviceSynchronize();

    // Get flux in y-direction
    soln_getFlux<<<grid,block>>>(dt, d_gr_vly, d_gr_vry, d_gr_fluxy, 1);
    CudaCheckError();
    cudaDeviceSynchronize();
 }
