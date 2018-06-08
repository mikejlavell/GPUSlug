/*CUDA Implementation of the Roe Riemann Solver 
Written by Steven Reeves, University of California, Santa Cruz
May 16th, 2017

2D Edits by Michael Lavell, UCSC
June 2018*/

/*------------------ Library Dependencies --------------------------------*/
#include <cuda.h>
#include <cmath>
#include "cuda_runtime.h"
#include <device_functions.h>

/*-----------------------Function Dependencies!----------------------*/
#include "primconsflux.cuh"
#include "eigensystem.cuh"
#include "definition.h"
__device__ double dot_product(double4 u, double4 v);



/*---------------------- ROE SOLVER --------------------------------*/
__device__ void roe(const double4 vL, const double4 vR, double4 &Flux, int dir)
{
  double4 FL={0.0},FR={0.0},uL={0.0},uR={0.0};
  double4 vAvg={0.0};
  double lambda[NUMB_WAVE]={0.0};
  double4 reig[NUMB_WAVE]={0.0}, leig[NUMB_WAVE]={0.0};
  int conservative = 1;
  double4 vec, sigma;
  
  // set the initial sum to be zero
  sigma = {0.0};
  vec = {0.0};
  
  //Calculate the average state
  vAvg.x = 0.5*(vL.x + vR.x);
  vAvg.y = 0.5*(vL.y + vR.y);
  vAvg.z = 0.5*(vL.z + vR.z);
  vAvg.w = 0.5*(vL.w + vR.w);

  //Find Average States and Eigenvalues + Eigenvectors for Averaged State  
  eigenvalues(vAvg,lambda,dir);
  left_eigenvectors(vAvg,conservative,leig,dir);
  right_eigenvectors(vAvg,conservative,reig,dir);
    
  //Fnding the Fluxes for the Aeraged State and Conservative Vars for Roe Flux.   
  prim2flux(vL,FL,dir);
  prim2flux(vR,FR,dir);
  prim2cons(vL,uL);
  prim2cons(vR,uR);

  vec.x = uR.x - uL.x;
  vec.y = uR.y - uL.y;
  vec.z = uR.z - uL.z;
  vec.w = uR.w = uL.w;

  for(int kWaveNum = 0; kWaveNum<NUMB_WAVE; kWaveNum++)
  {
     sigma.x += dot_product(leig[kWaveNum],vec)*fabs(lambda[kWaveNum])*reig[kWaveNum].x;
     sigma.y += dot_product(leig[kWaveNum],vec)*fabs(lambda[kWaveNum])*reig[kWaveNum].y;
     sigma.z += dot_product(leig[kWaveNum],vec)*fabs(lambda[kWaveNum])*reig[kWaveNum].z;
     sigma.w += dot_product(leig[kWaveNum],vec)*fabs(lambda[kWaveNum])*reig[kWaveNum].w;
  }
  
  // numerical flux
  Flux.x = 0.5*(FL.x + FR.x) - 0.5*sigma.x; //Density
  Flux.y = 0.5*(FL.y + FR.y) - 0.5*sigma.y; //X-Momentum/X-Velocity
  Flux.z = 0.5*(FL.z + FR.z) - 0.5*sigma.z; //Y-Momentum/Y-Velocity
  Flux.w = 0.5*(FL.w + FR.w) - 0.5*sigma.w; //Energy/Pression
}
