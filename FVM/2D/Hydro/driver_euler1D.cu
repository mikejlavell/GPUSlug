/*CUDA Implementation of the 1D Slug Code, for limited spatial reconstructions! 
Written by Steven Reeves, University of California, Santa Cruz
May 10th, 2017

2D Edits by Michael Lavell, UCSC
May, 2018

*/

/*-----------------------Library Dependencies!----------------------*/
#include <iostream>
#include <cuda.h>
#include <fstream>
#include <cmath>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <time.h>

/*-----------------------Header Dependencies!----------------------*/
#include "Slug_helper.cuh"
#include "cfl.cuh"
#include "soln_ReconEvolve.cuh"
#include "io_helper.h"

extern double sim_tmax;
/*---------------Main Program, calls the routines!------*/
int main(){
	//Useful Debugging Features
		 cudaError_t cudaStatus;

 // Choose which GPU to run on, change this on a multi-GPU system.
 cudaStatus = cudaSetDevice(0);
 if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaSetDevice failed!");
 }
 // Grid and Block size for CUDA Kernels
  dim3 block(BLOCK_DIMX,BLOCK_DIMY,1);
  dim3 grid(ceil(GRID_XSZE/block.x), ceil(GRID_YSZE/block.y , 1);
  double t,dt;
  int nStep,ioCounter,zerodt;
  double ioCheckTime;

  zerodt = 1;
  t = 0.0;

  nStep = 0;
  ioCounter = 0;
  
 /* -------------------- Grid Initialization ------------------------------*/

  /*-------------Host Variables ----------------------*/
  double *gr_xCoord; //Coordinate System
  double *gr_yCoord; 
  double4* gr_U; // conservative vars
  double4* gr_V; // primitive vars

  /*-----------------------Device Variables ----------------------------------*/
  double4* d_gr_U; // conservative vars
  double4* d_gr_V; // primitive vars

  double4* d_gr_vLX;   // left Riemann states in x-direction
  double4* d_gr_vRX;   // right Riemann states in x-direction
  double4* d_gr_fluxX; // fluxes in x-direction

  double4* d_gr_vLY;   // left Riemann states in y-direction
  double4* d_gr_vRY;   // right Riemann states in y-direction
  double4* d_gr_fluxY; // fluxes in y-direction


 /*--------------------- Allocate -------------------------------------------*/
  //Grid Variables
  grid_alloc(gr_xCoord, gr_yCoord, gr_U, gr_V, //Host
  d_gr_U, d_gr_V, d_gr_vLX, d_gr_vRX, d_gr_fluxX, d_gr_vLY, d_gr_vRY, d_gr_fluxY); //Device

 /*-------------------  Simulation Initialization  -----------------------------*/
  sim_init(gr_V, gr_U, gr_xCoord, gr_yCoord);
  //Write Initial Condition
  io_writeOutput(ioCounter, gr_xCoord, gr_yCoord, gr_V);
  ioCounter += 1;
  clock_t tStart = clock();

/* =========================== Simulate =========================================*/
 while (t < sim_tmax){
	 //calculate time step
	 //dt = cfl_cuda(gr_V);
	 dt = cfl_omp(gr_V);

     if ( fabs(t - sim_tmax) <= dt ){
        dt = fabs(t - sim_tmax);
      }
      
     //Transfer to GPU
     transfer_to_gpu(gr_V, gr_U, d_gr_V, d_gr_U); 
     
 /*------------------  Reconstruct and Update  --------------------------------*/
     // Launches reconstruction reconstruction and flux kernel in x and y 
     soln_ReconEvolveAvg( dt, d_gr_V, d_gr_U, d_gr_vLX, d_gr_vRX, d_gr_fluxX,
			  d_gr_vLY, d_gr_vRY, d_gr_fluxY, grid, block);
 
     //  Update the solution. 
     soln_update<<<grid,block>>>(d_gr_U, d_gr_V, d_gr_fluxX, d_gr_fluxY, dt); 
     CudaCheckError();
     cudaDeviceSynchronize();

     //Transfer to CPU
     transfer_to_cpu(d_gr_V, d_gr_U, gr_V, gr_U);   

     //call BC on Primitive vars
     bc_apply(gr_V);
     //Call BC on Conservative vars
     bc_apply(gr_U);
     
     
     //update your time and step count
     t += dt;
     nStep += 1;

     if (dt <= 0.0) break;
 }

/*------------------- Write End-Time Solution ------------------------------*/
  io_writeOutput(ioCounter, gr_xCoord, gr_yCoord, gr_V);

/*--------------------- Free the Variables ------------------------------------*/
  grid_finalize(gr_xCoord ,gr_U, gr_V, //Host
  d_gr_U, d_gr_V, d_gr_vLX, d_gr_vRX, d_gr_fluxX, d_gr_vLY, d_gr_vRY, d_gr_fluxY); //Device
}
