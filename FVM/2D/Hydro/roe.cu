/*CUDA Implementation of the Roe Riemann Solver 
Written by Steven Reeves, University of California, Santa Cruz
May 16th, 2017*/

/*------------------ Library Dependencies --------------------------------*/
#include <cuda.h>
#include <cmath>
#include "cuda_runtime.h"
#include <device_functions.h>

/*-----------------------Function Dependencies!----------------------*/
#include "primconsflux.cuh"
#include "eigensystem.cuh"

/*---------------------- ROE SOLVER --------------------------------*/
__device__ void roe(double3 *vL,double3 *vR,double3 *Flux,)
{
  double3 FL,FR,uL,uR;
  double3 vAvg;
  double lambda[NUMB_WAVE];
  double dot_product;
  double3 reig[NUMB_WAVE], leig[NUMB_WAVE];
  bool conservative;
  double3 vec, sigma;
  int kWaveNum;
  
  // set the initial sum to be zero
  sigma = {0.0, 0.0, 0.0};
  vec = {0.0, 0.0, 0.0};
  
  // we need conservative eigenvectors
  conservative = 1;

//Calculate the average state
vAvg.x = 0.5*(vL.x + vR.x);
vAvg.y = 0.5*(vL.y + vR.y);
vAvg.z = 0.5*(vL.z + vR.z);

//Find Average States and Eigenvalues + Eigenvectors for Averaged State  
    eigenvalues(vAvg,lambda);
    left_eigenvectors(vAvg,conservative,leig);
    right_eigenvectors(vAvg,conservative,reig);
    
//Fnding the Fluxes for the Aeraged State and Conservative Vars for Roe Flux.   
    prim2flux(vL,FL);
    prim2flux(vR,FR);
    prim2cons(vL,uL);
    prim2cons(vR,uR);

    vec.x = uR.x - uL.x;
    vec.y = uR.y - uL.y;
    vec.z = uR.z - uL.z;
  for(kWaveNum = 1,kWaveNum<NUMB_WAVE,kWaveNum++)
  {
     dot_product = leig.x[kWaveNum]*vec.x + leig.y[kWaveNum]*vec.y + leig.z[kWaveNum]*vec.z;
     sigma.x += dot_product*abs(lambda(kWaveNum))*reig.x[kWaveNum];
     sigma.y += dot_product*abs(lambda(kWaveNum))*reig.y[kWaveNum];
     sigma.z += dot_product*abs(lambda(kWaveNum))*reig.z[kWaveNum];
    }
  
  // numerical flux
  Flux.x = 0.5*(FL.x + FR.x) - 0.5*sigma.x; //Density
  Flux.y = 0.5*(FL.y + FR.y) - 0.5*sigma.y; //Momentum/Velocity
  Flux.z = 0.5*(FL.z + FR.z) - 0.5*sigma.z; //Energy/Pressure
}
