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
extern int GRID_SIZE;
extern int gr_ngc;


 __device__ double dot_product(double3 u, double3 v){
    double ans = 0.0;
    ans = u.x*v.x + u.y*v.y + u.z*v.z;
    return ans;
 }
 
/*------------------------------ Reconstruction Kernel ----------------------------*/
 __global__ void soln_reconstruct_PLM(double dt, double3 *V, double3 *vl,
  double3 *vr)
 {
    int tid = threadIdx.x + blockIdx.x*blockDim.x + gr_ngc;
    if(tid < GRID_SIZE - gr_ngc)
    {
    double lambda[NUMB_WAVE] = {0.0};
    double3 leig[NUMB_WAVE];
    double3 reig[NUMB_WAVE];
    double3 delL, delR; 
    double pL, pR, delW[NUMB_WAVE];
    eigenvalues(V[tid],lambda); //Get Eigenvalues for reconstruction
    left_eigenvectors (V[tid],conservative,leig);
    right_eigenvectors(V[tid],conservative,reig);

        for(int kWaveNum = 1; kWaveNum < NUMB_WAVE; kWaveNum ++)
        {
           delL = V[tid]-V[tid-1];
           delR = V[tid+1]-V[tid];
           // project onto characteristic vars
           pL = dot_product(leig[kWaveNum], delL);
           pR = dot_product(leig[kWaveNum], delR);
           // Use a TVD Slope limiter
           if (sim_limiter == "minmod"){
                minmod(pL, pR, delW[kWaveNum]):
              }
           else if (sim_limiter == "vanLeer"){
                vanLeer(pL, pR, delW[kWaveNum]);
              }
           else if (sim_limiter == "mc"){
                mc(pL, pR, delW[kWaveNum]);
              }
        }   
        
         //do char tracing
        //set the initial sum to be zero
       double3 sigL= {0.0,0.0,0.0};
       double3 sigR= {0.0,0.0,0.0};

        if (sim_riemann == "roe"){
	        for(int kWaveNum = 0; kWaveNum<NUMB_WAVE; kWaveNum++){
		        lambdaDtDx = lambda[kWaveNum]*dt/gr_dx;
              if (lambda(kWaveNum) > 0){
              //Right Sum
                 sigR.x += 0.5*(1.0 - lambdaDtDx)*reig.x[kWaveNum]*delW[kWaveNum];
        		 sigR.y += 0.5*(1.0 - lambdaDtDx)*reig.y[kWaveNum]*delW[kWaveNum];
		         sigR.z += 0.5*(1.0 - lambdaDtDx)*reig.z[kWaveNum]*delW[kWaveNum];
		        }
              else if (lambda(kWaveNum) < 0){
              //Left Sum
                 sigL.x += 0.5*(-1.0 - lambdaDtDx)*reig.x[kWaveNum]*delW[kWaveNum];
                 sigL.y += 0.5*(-1.0 - lambdaDtDx)*reig.y[kWaveNum]*delW[kWaveNum];
                 sigL.z += 0.5*(-1.0 - lambdaDtDx)*reig.z[kWaveNum]*delW[kWaveNum];
        		}
	        }
         }
      else if (sim_riemann == "hll" || sim_riemann == "hllc")
	     {
	       for(int kWaveNum = 0; kWaveNum<NUMB_WAVE; kWaveNum++)
		    {
		     lambdaDtDx = lambda[kWaveNum]*dt/gr_dx;
            //Right Sum
             sigR.x += 0.5*(1.0 - lambdaDtDx)*reig.x[kWaveNum]*delW[kWaveNum];
	     sigR.y += 0.5*(1.0 - lambdaDtDx)*reig.y[kWaveNum]*delW[kWaveNum];
	     sigR.z += 0.5*(1.0 - lambdaDtDx)*reig.z[kWaveNum]*delW[kWaveNum];
           //Left Sum
             sigL.x += 0.5*(-1.0 - lambdaDtDx)*reig.x[kWaveNum]*delW[kWaveNum];
             sigL.y += 0.5*(-1.0 - lambdaDtDx)*reig.y[kWaveNum]*delW[kWaveNum];
             sigL.z += 0.5*(-1.0 - lambdaDtDx)*reig.z[kWaveNum]*delW[kWaveNum];
	    	}     	
	    }

           // Now PLM reconstruction for dens, velx, and pres
           gr_vL.x[tid] = V.x[tid] + sigL.x;
           gr_vL.y[tid] = V.y[tid] + sigL.y;
           gr_vL.z[tid] = V.z[tid] + sigL.z;
           gr_vR.x[tid] = V.x[tid] + sigR.x;
           gr_vR.y[tid] = V.y[tid] + sigR.y;
           gr_vR.z[tid] = V.z[tid] + sigR.z;   
     }
 }

/*---------------------------------- Get Flux Kernel ---------------------------*/ 
__global__ void soln_getFlux(double dt, double3 *vl, double3 *vr, double3 *U,
 				double3 *flux)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x + gr_ngc;
	if(tid < GRID_SIZE - gr_ngc) 
	{
/*		if(sim_riemann == "hll")
		{
			hll(vR[tid-1],vL[tid],flux[tid]);//Call the HLL Riemann Solver
		}
		if(sim_riemann == "hllc")
		{
			hllc(vR[tid-1],vL[tid],flux[tid]);//Call the HLLC Riemann Solver
		}//*/
		if(sim_riemann == "roe")
		{
			roe(vR[tid-1],vL[tid],flux[tid]);//Call the Roe Riemann Solver
		}
	}
}

/*-------------------  Solution Reconstruct Evolve and Average -----------------*/
void soln_ReconEvolveAvg(double dt, double3 *d_gr_V, double3 *d_gr_U,
 double3 * d_gr_vl, double3 *d_gr_vr, double3 *d_gr_flux, dim3 grid, dim3 block) 
 {
    soln_reconstruct_PLM<<<grid,block>>>(dt, d_gr_V, d_gr_vl, d_gr_vr);
    CudaCheckError();
    cudaDeviceSynchronize();

    soln_getFlux<<<grid,block>>>(dt, d_gr_vl, d_gr_vr, d_gr_U, d_gr_flux);
    CudaCheckError();
    cudaDeviceSynchronize();
 }
