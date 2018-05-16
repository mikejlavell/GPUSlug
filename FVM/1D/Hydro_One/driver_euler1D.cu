/*CUDA Implementation of the 1D Slug Code, for limited spatial reconstructions! 
Written by Steven Reeves, University of California, Santa Cruz
May 10th, 2017
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
  dim3 block(512,1,1);
  dim3 grid(ceil(GRID_SIZE/block.x), 1 , 1);
  double t,dt;
  int nStep,ioCounter;

  t = 0.0;

  nStep = 0;
  ioCounter = 0;
  
 /* -------------------- Grid Initialization ------------------------------*/

  /*-------------Host Variables ----------------------*/
  double *gr_xCoord; //Coordinate System
  double3* gr_U; // conservative vars
  double3* gr_V; // primitive vars

  /*-----------------------Device Variables ----------------------------------*/
  double3* d_gr_U; // conservative vars
  double3* d_gr_V; // primitive vars

  double3* d_gr_vL;   // left Riemann states
  double3* d_gr_vR;   // right Riemann states
  double3* d_gr_flux; // fluxes

 /*--------------------- Allocate -------------------------------------------*/
  //Grid Variables
 // grid_alloc(gr_xCoord ,gr_U, gr_V, //Host
 // d_gr_U, d_gr_V, d_gr_vL, d_gr_vR, d_gr_flux); //Device
 size_t num_bytes = GRID_SIZE*sizeof(double3);  
 size_t num_doub = GRID_SIZE*sizeof(double); 
 //Allocating Host Variables
 gr_xCoord = (double*)malloc(num_doub);
 gr_U = (double3*)malloc(num_bytes);   
 gr_V = (double3*)malloc(num_bytes);
 
 
 //Allocating Device Variables
 CudaSafeCall(cudaMalloc((void**)&d_gr_U, num_bytes));
 CudaSafeCall(cudaMalloc((void**)&d_gr_V, num_bytes));
 CudaSafeCall(cudaMalloc((void**)&d_gr_vR, num_bytes));
 CudaSafeCall(cudaMalloc((void**)&d_gr_vL, num_bytes));
 CudaSafeCall(cudaMalloc((void**)&d_gr_flux, num_bytes));
 cudaMemset(d_gr_U, 0.0, num_bytes);
 cudaMemset(d_gr_V, 0.0, num_bytes);
 cudaMemset(d_gr_vL, 0.0, num_bytes);
 cudaMemset(d_gr_vR, 0.0, num_bytes);
 cudaMemset(d_gr_flux, 0.0, num_bytes);
 /*-------------------  Simulation Initialization  -----------------------------*/
  sim_init(gr_V, gr_U, gr_xCoord);
  //Write Initial Condition
  io_writeOutput(ioCounter, gr_xCoord, gr_V);
  ioCounter += 1;
	clock_t tStart = clock();

/* =========================== Simulate =========================================*/
while (t < sim_tmax){
	 //calculate time step
	 dt = cfl(gr_V);
     if ( fabs(t - sim_tmax) <= dt ){
        dt = fabs(t - sim_tmax);
      }
      
     //Transfer to GPU
	 transfer_to_gpu(gr_V, gr_U, d_gr_V, d_gr_U); 
 /*------------------  Reconstruct and Update  --------------------------------*/
    //Launches Kernel to reconstruct cell interface values in d_gr_vL and vR respectively
     soln_reconstruct_PLM<<<grid,block>>>(dt, d_gr_V, d_gr_vL, d_gr_vR);
    CudaCheckError();
    cudaDeviceSynchronize();
    // And gets the Numerical Flux. 
    soln_getFlux<<<grid,block>>>(dt, d_gr_vL, d_gr_vR, d_gr_flux);
    CudaCheckError();
    cudaDeviceSynchronize();
    //  Updates the solution. 
     soln_update<<<grid,block>>>(d_gr_U, d_gr_V, d_gr_flux, dt);  
    	CudaCheckError();
    	cudaDeviceSynchronize();//*/
    //Launches Kernel to update solution.
     
     //Transfer to CPU
     transfer_to_cpu(d_gr_V, d_gr_U, gr_V, gr_U);  
     //call BC on Primitive vars
     bc_apply(gr_V);
     //Call BC on Conservative vars
     bc_apply(gr_U);
    //update your time and step count
     t += dt;
     nStep += 1;
	if(dt < 0.0) break;
	//if(nStep > 1) break;
}

printf("Time taken: %fs\n", (double)(clock() - tStart) / (ncores*CLOCKS_PER_SEC));
/*------------------- Write End-Time Solution ------------------------------*/
  io_writeOutput(ioCounter, gr_xCoord, gr_V);

/*--------------------- Free the Variables ------------------------------------*/
   //Deallocating Host Variables
 free(gr_xCoord);
 free(gr_U);
 free(gr_V);
 
 //Deallocating Device Variables
 cudaFree(d_gr_U);
 cudaFree(d_gr_V);
 cudaFree(d_gr_vR);
 cudaFree(d_gr_vL);
 cudaFree(d_gr_flux);
}
