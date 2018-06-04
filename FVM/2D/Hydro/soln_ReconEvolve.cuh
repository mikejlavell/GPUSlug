/*------------------------------- Library Dependencies ---------------------------*/
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>
#include <algorithm>
 
/*------------------------------ Program Dependencies ----------------------------*/
#include "eigensystem.cuh"
extern string sim_riemann;
extern int NUMB_WAVE;
extern int GRID_XSZE;
extern int GRID_YSZE;
extern int gr_ngc;


 __device__ double dot_product(double4 u, double4 v){
    double ans = 0.0;
    ans = u.x*v.x + u.y*v.y + u.z*v.z u.w*v.w;
    return ans;
 }
 
/*------------------------------ Reconstruction Kernel ----------------------------*/
 __global__ void soln_reconstruct_PLM(double dt, double4 *V, double4 *vl,
  double4 *vr, int dir)
 {
    if( dir == 0 ){
	int GRID_SIZE = GRID_XSZE;
	double gr_dX = gr_dx;
	}
    else if( dir == 1){
	int GRID_SIZE = GRID_YSZE;
	double gr_dX = gr_dy;
	}

    int tid = threadIdx.x + blockIdx.x*blockDim.x + gr_ngc;
    if(tid < GRID_SIZE - gr_ngc)
    {
    double lambda[NUMB_WAVE] = {0.0};
    double4 leig[NUMB_WAVE] = {0.0};
    double4 reig[NUMB_WAVE] = {0.0};
    double4 delL = {0.0}, delR= {0.0}; 
    double pL = 0.0, pR = 0.0, delW[NUMB_WAVE] = {0.0};

    // Calculate eigensystem
    eigenvalues(V[tid],lambda); //Get Eigenvalues for reconstruction
    left_eigenvectors (V[tid],conservative,leig);
    right_eigenvectors(V[tid],conservative,reig);
    __syncthreads();

        for(int kWaveNum = 0; kWaveNum < NUMB_WAVE; kWaveNum ++)
        {
           delL.x = V[tid].x-V[tid-1].x;
           delL.y = V[tid].y-V[tid-1].y;
           delL.z = V[tid].z-V[tid-1].z;
           delL.w = V[tid].w-V[tid-1].w;
           delR.x = V[tid+1].x-V[tid].x;
           delR.y = V[tid+1].y-V[tid].y;
           delR.z = V[tid+1].z-V[tid].z;
           delR.w = V[tid+1].w-V[tid].w;
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
		        }
              else if (lambda(kWaveNum) < 0){
              //Left Sum
                 sigL.x += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].x*delW[kWaveNum];
                 sigL.y += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].y*delW[kWaveNum];
                 sigL.z += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].z*delW[kWaveNum];
                 sigL.w += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].w*delW[kWaveNum];
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
	    	}     	
	    }

           // Now PLM reconstruction for dens, velx, vely, and pres
           vl[tid].x = V[tid].x + sigL.x;
           vl[tid].y = V[tid].y + sigL.y;
           vl[tid].z = V[tid].z + sigL.z;
           vl[tid].w = V[tid].w + sigL.w;
           vr[tid].x = V[tid].x + sigR.x;
           vr[tid].y = V[tid].y + sigR.y;
           vr[tid].z = V[tid].z + sigR.z;   
           vr[tid].w = V[tid].w + sigR.w;   
	   }
     }
 }

/*---------------------------------- Get Flux Kernel ---------------------------*/ 
__global__ void soln_getFlux(double dt, double4 *vl, double4 *vr, double4 *flux, int dir)
{
	if( dir == 0)
		GRID_SIZE = GRID_XSZE;
	else if( dir == 1)
		GRID_SIZE = GRID_YSZE;

	int tid = threadIdx.x + blockIdx.x*blockDim.x + gr_ngc;
	if(tid < GRID_SIZE - gr_ngc) 
	{
/*		if(sim_riemann == "hll")
		{
			hll(vr[tid-1],vl[tid],flux[tid]); //Call the HLL Riemann Solver
		}
		if(sim_riemann == "hllc")
		{
			hllc(vr[tid-1],vl[tid],flux[tid]); //Call the HLLC Riemann Solver
		}//*/
		if(sim_riemann == "roe")
		{
			roe(vr[tid-1],vl[tid],flux[tid]); //Call the Roe Riemann Solver
		}
	}
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
    soln_getFlux<<<grid,block>>>(dt, d_gr_vly, d_gr_vry, d_gr_U, d_gr_fluxx, 0)
    CudaCheckError();
    cudaDeviceSynchronize();

    // Reconstruct in y-direction
    soln_reconstruct_PLM<<<grid,block>>>(dt, d_gr_V, d_gr_vly, d_gr_vry, 1);
    CudaCheckError();
    cudaDeviceSynchronize();

    // Get flux in y-direction
    soln_getFlux<<<grid,block>>>(dt, d_gr_vly, d_gr_vry, d_gr_U, d_gr_fluxy, 1);
    CudaCheckError();
    cudaDeviceSynchronize();
 }
