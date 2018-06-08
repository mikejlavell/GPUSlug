/*------------------------------- Library Dependencies ---------------------------*/
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>
#include <algorithm>
 
/*------------------------------ Program Dependencies ----------------------------*/
#include "definition.h"
#include "slope_limiter.cuh"
#include "eigensystem.cuh" 
#include "Slug_helper.cuh"

extern __device__ void roe(const double4 vL, const double4 vR, double4 &Flux, int dir);

__device__ double dot_product(double4 u, double4 v){
    double ans = 0.0;
    ans = u.x*v.x + u.y*v.y + u.z*v.z + u.w*v.w;
    return ans;
 }
 
/*------------------------------ Reconstruction Kernel ----------------------------*/
 __global__ void soln_reconstruct_PLM(const double dt, double4 *V, double4 *vl,
  double4 *vr, int dir)
 {
    int a, b;
    double gr_dX;
    if( dir == 0 ){
	//int GRID_SIZE = GRID_XSZE;
	gr_dX = gr_dx;
	a = 1; b = 0;
	}
    else if( dir == 1){
	//int GRID_SIZE = GRID_YSZE;
	gr_dX = gr_dy;
	a = 0; b = 1;
	}

    int tidx = threadIdx.x + blockIdx.x*blockDim.x + gr_ngc;
    int tidy = threadIdx.y + blockIdx.y*blockDim.y + gr_ngc;
    int stid = tidy*M + tidx;
    int stidm = (tidy-b)*M + tidx - a;
    int stidp = (tidy+b)*M + tidx + a;


    if(tidx < GRID_XSZE - gr_ngc){
    if(tidy < GRID_YSZE - gr_ngc){
    double lambdaDtDx;
    double lambda[NUMB_WAVE] = {0.0};
    double4 leig[NUMB_WAVE] = {0.0};
    double4 reig[NUMB_WAVE] = {0.0};
    double4 delL = {0.0}, delR= {0.0}; 
    double pL = 0.0, pR = 0.0, delW[NUMB_WAVE] = {0.0};

    // Calculate eigensystem
    eigenvalues(V[stid],lambda,dir); //Get Eigenvalues for reconstruction
    left_eigenvectors (V[stid],0,leig,dir);
    right_eigenvectors(V[stid],0,reig,dir);
    __syncthreads();

        for(int kWaveNum = 0; kWaveNum < NUMB_WAVE; kWaveNum ++)
        {
           delL.x = V[stid].x -V[stidm].x;
           delL.y = V[stid].y -V[stidm].y;
           delL.z = V[stid].z -V[stidm].z;
           delL.w = V[stid].w -V[stidm].w;
           delR.x = V[stidp].x-V[stid].x;
           delR.y = V[stidp].y-V[stid].y;
           delR.z = V[stidp].z-V[stid].z;
           delR.w = V[stidp].w-V[stid].w;
	   __syncthreads();

           // project onto characteristic vars
           pL = dot_product(leig[kWaveNum], delL);
           pR = dot_product(leig[kWaveNum], delR);

           // Use a TVD Slope limiter
           if (sim_limiter == 1){
                delW[kWaveNum] = minmod(pL, pR);
              }
	   /*
           else if (sim_limiter == 2){
                delW[kWaveNum] = vanLeer(pL, pR);
              }
           else if (sim_limiter == 3){
                delW[kWaveNum] = mc(pL, pR);
              }*/
	   __syncthreads();
        }   
        
       //do char tracing
       //set the initial sum to be zero
       double4 sigL= {0.0,0.0,0.0,0.0};
       double4 sigR= {0.0,0.0,0.0,0.0};

        if (sim_riemann == "roe"){
	   for(int kWaveNum = 0; kWaveNum<NUMB_WAVE; kWaveNum++){
	      lambdaDtDx = lambda[kWaveNum]*dt/gr_dX;
              if (lambda[kWaveNum] > 0){
              //Right Sum
                 sigR.x += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].x*delW[kWaveNum];
        	 sigR.y += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].y*delW[kWaveNum];
		 sigR.z += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].z*delW[kWaveNum];
		 sigR.w += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].w*delW[kWaveNum];
		 __syncthreads();        
	      }
              else if (lambda[kWaveNum] < 0){
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
         vl[stid].x = V[stid].x + sigL.x;
         vl[stid].y = V[stid].y + sigL.y;
         vl[stid].z = V[stid].z + sigL.z;
         vl[stid].w = V[stid].w + sigL.w;
         vr[stid].x = V[stid].x + sigR.x;
         vr[stid].y = V[stid].y + sigR.y;
         vr[stid].z = V[stid].z + sigR.z;   
         vr[stid].w = V[stid].w + sigR.w;   
	 }
     }
 }

/*---------------------------------- Get Flux Kernel ---------------------------*/ 
__global__ void soln_getFlux(double dt, double4 *vl, double4 *vr, double4 *flux, int dir)
{
	int a, b;

	if( dir == 0) a = 1, b = 0;
	else if( dir == 1) a = 0, b = 1;

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
			roe(vr[(tidy-b)*M + tidx -a],vl[tidy*M + tidx],flux[tidy*M + tidx],dir); 
		}
	}}
}

/*-------------------  Solution Reconstruct Evolve and Average -----------------*/
void soln_ReconEvolveAvg(double dt, double4 *d_gr_V, double4 *d_gr_U,
 double4 * d_gr_vlx, double4 *d_gr_vrx, double4 *d_gr_fluxx,
 double4 * d_gr_vly, double4 *d_gr_vry, double4 *d_gr_fluxy, const dim3 grid, const dim3 block) 
 {
	
    // Reconstruct in x-direction
    soln_reconstruct_PLM<<<mygrid,myblock,0>>>(dt, d_gr_V, d_gr_vlx, d_gr_vrx, 0);
	CudaCheckError(); 
    // Get flux in x-direction
    soln_getFlux<<<grid,block>>>(dt, d_gr_vly, d_gr_vry, d_gr_fluxx, 0);
	CudaCheckError();
    // Reconstruct in y-direction
    soln_reconstruct_PLM<<<grid,block>>>(dt, d_gr_V, d_gr_vly, d_gr_vry, 1);
	CudaCheckError();
    // Get flux in y-direction
    soln_getFlux<<<grid,block>>>(dt, d_gr_vly, d_gr_vry, d_gr_fluxy, 1);
	CudaCheckError();
} 

