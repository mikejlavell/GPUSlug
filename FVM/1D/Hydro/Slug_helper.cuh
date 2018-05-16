#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>

#include "definition.h"

extern int GRID_SIZE;
extern int gr_dx;
extern int gr_iend;
extern int gr_ibeg;
extern int gr_imax;
extern int gr_ngc;
extern int sim_bcType;
extern string sim_riemann;

/*----------------------CUDA Error stuff -------------------------------- */ 
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

/*--------------------- Allocate Variables ----------------------------------*/
void grid_alloc( double *gr_xCoord,
 double3 *gr_U, double3 *gr_V, double3 *gr_vL, double3 *gr_vR, double3 *gr_flux, //Host
 double3 *d_gr_U, double3 *d_gr_V, double3 *d_gr_vL, double3 *d_gr_vR, double3 *d_gr_flux) //Device
{
 size_t num_bytes = GRID_SIZE*sizeof(double3*);   
 //Allocating Host Variables
 gr_xCoord = (double*)malloc(GRID_SIZE*sizeof(double));
 gr_U = (double3*)malloc(num_bytes);   
 gr_V = (double3*)malloc(num_bytes);
 gr_vR = (double3*)malloc(num_bytes);
 gr_vL = (double3*)malloc(num_bytes);  
 gr_flux = (double3*)malloc(num_bytes);
 
 //Allocating Device Variables
 CudaSafeCall(cudaMalloc((void**)&d_gr_U, num_bytes));
 CudaSafeCall(cudaMalloc((void**)&d_gr_V, num_bytes));
 CudaSafeCall(cudaMalloc((void**)&d_gr_vR, num_bytes));
 CudaSafeCall(cudaMalloc((void**)&d_gr_vL, num_bytes));
 CudaSafeCall(cudaMalloc((void**)&d_gr_flux, num_bytes));
}


/*----------------------- Initial Conditions ----------------------------------*/
void sim_init(double3 *gr_V, double3 *gr_U, double *gr_xCoord)
{
    //initialize grid coordinates
#pragma omp parallel for num_threads(ncores)    
    for(int i = 0; i < GRID_SIZE; i ++ ) gr_xCoord[i] = i*gr_dx;
    if(sim_type == 1) // Sod's Shock Tube
     {
        for(int i = 0; i < GRID_SIZE; i++)
            {
                if (gr_xCoord[i] < sim_shockLoc)
                {
                   gr_V.x[i] = sim_densL;
                   gr_V.y[i] = sim_velxL;
                   gr_V.z[i] = sim_presL;
                else
                {
                   gr_V.x[i] = sim_densR;
                   gr_V.y[i] = sim_velxR;
                   gr_V.z[i] = sim_presR;
                }   
        }
     }
           
    else if(sim_type == 2) // Rarefaction
    
    else if(sim_type == 3) // Blast2
    {
#pragma omp parallel for num_threads(ncores)    
        for(int i = 0; i<GRID_SIZE; i++) 
        {
            if (gr_xCoord[i] <= 0.1)
               {
                   gr_V.x[i] = 1.0;
                   gr_V.y[i] = 0.0;
                   gr_V.z[i] = 1000.0;
               }
            else if (gr_xCoord[i] <= 0.9 && gr_xCoord[i] > 0.1)
              {
               //middle state
                   gr_V.x[i] = 1.0;
                   gr_V.y[i] = 0.0;
                   gr_V.z[i] = 0.01;
              }
            else
              {
                   gr_V.x[i] = 1.0;
                   gr_V.y[i] = 0.0;
                   gr_V.z[i] = 100.0;
             }
         }
      } 
    else if(sim_type == 4) // Shu-Osher
    {
        double x; 
        //IC for shu-osher problem
        //transform the domain [0,1] onto the one give for the problem:[-4.5,4.5]
#pragma omp parallel for num_threads(ncores) private(x)        
       for(int i = 0; i < GRID_SIZE; i++)
       {
            x = gr_xCoord[i] - 4.5;
            if (x < -4.0)
            {
            //left state
                gr_V.x[i] = 3.857143;
                gr_V.y[i] = 2.629369;
                gr_V.z[i] = 10.33333;
            }   
        else
        {
           //Right State
                gr_V.x[i] = 1 + 0.2*sin(5.0*x);
                gr_V.y[i] = 0.0;
                gr_V.z[i] = 1.0;
            }
        }
    }
#pragma omp parallel for num_threads(ncores)    
    for(int i = 0; i < GRID_SIZE; i++) prim2cons(gr_V[i], gr_U[i]);
}

//------------------------- CFL Condition ------------------------------------
double cfl(double3 *gr_V)
{   
    double cs, lambda, maxSpeed, dt; 
    maxSpeed = -1e30;
#pragma omp parallel for reduction(max:maxSpeed) num_threads(ncores) private(cs, lambda)
 for(int i = gr_ibeg, gr_iend)
    {
            
         cs = sqrtf(sim_gamma*gr_V.z[i]/gr_V.x[i]);
         lambda=(abs(gr_V.y[i]) + cs);
         maxSpeed=max(maxSpeed,lambda);
     }

  // cfl
  dt = 0.8*gr_dx/maxSpeed;
  return dt;
}
//------------------------- Boundary Condition Operators ---------------------

void bc_apply(double3 *U)
{
    if (sim_bcType == 1)
    { // Outflow
       bc_outflow(U);
    }
    else if (sim_bcType == 2) //User 
    {
        //Do nothing, it's for the Shu-Osher Problem
    }   
    else if (sim_bcType == 3) //Reflect
    {
       bc_reflect(U);
    }   
    else if (sim_bcType == 4) //Periodic
    {
       bc_periodic(U);
    }
}

void bc_outflow(double3 *V)
{
    for(int i = 0; i < gr_ngc ; i++)
    {
       // on the left GC
       V.x[i] = V.x[i+1];
       V.y[i] = V.y[i+1];
       V.z[i] = V.z[i+1];       
       // on the right GC
       V.x[gr_imax+1-i] = V.x[gr_imax-i];
       V.y[gr_imax+1-i] = V.y[gr_imax-i];
       V.z[gr_imax+1-i] = V.z[gr_imax-i];              
    }
}

void bc_reflect(double3 *V)
{
   for( int i = 0, i < gr_ngc; i++)
       int k0 = 2*gr_ngc+1;
       int k1 = gr_iend-gr_ngc;

       // on the left GC
       V.x[i] = V.x[k0-i];
       V.y[i] =-V.y[k0-i];
       V.z[i] = V.z[k0-i];

       // on the right GC
       V.x[k1+k0-i] = V.x[k1+i];
       V.y[k1+k0-i] =-V.y[k1+i];
       V.z[k1+k0-i] = V.z[k1+i];
}
}

void bc_periodic(double3 *V)
{
     for(int i = 0, i < gr_ngc; i++)
     {  
       V.x[gr_iend+i] = V.x[ gr_ngc + i];
       V.y[gr_iend+i] = V.y[ gr_ngc + i];
       V.z[gr_iend+i] = V.z[ gr_ngc + i];
       
       V.x[gr_ibeg-i] = V.x[gr_iend - i+1];
       V.y[gr_ibeg-i] = V.y[gr_iend - i+1];
       V.z[gr_ibeg-i] = V.z[gr_iend - i+1];       
     }
}

/*----------------- Finalize the grid ---------------------------------------*/

void grid_finalize( double *gr_xCoord,
 double3 *gr_U, double3 *gr_V, double3 *gr_vL, double3 *gr_vR, double3 *gr_flux, //Host
 double3 *d_gr_U, double3 *d_gr_V, double3 *d_gr_vL, double3 *d_gr_vR, double3 *d_gr_flux) //Device
{
 //Deallocating Host Variables
 free(gr_xCoord);
 free(gr_U);
 free(gr_V);
 free(gr_vR);
 free(gr_vL);
 free(gr_flux);
 
 //Deallocating Device Variables
 cudaFree(d_gr_U);
 cudaFree(d_gr_V);
 cudaFree(d_gr_vR);
 cudaFree(d_gr_vL);
 cudaFree(d_gr_flux);
}


/*----------------- Transfer Routines ------------------------------------- */ 
void transfer_to_cpu(double3 *d_V, double3 *d_U, double3 *V, double3 *U) //gpu to cpu
{
   size_t num_bytes = GRID_SIZE*sizeof(double3);
   CudaSafeCall(cudaMemcpy(V, d_V, num_bytes, cudaMemcpyDeviceToHost));
   CudaSafeCall(cudaMemcpy(U, d_U, num_bytes, cudaMemcpyDeviceToHost));
   CudaCheckError();
   cudaDeviceSynchronize();
}

void transfer_to_gpu(double3 *V, double3 *U, double3 *d_V, double3 *d_U) //cpu to gpu
{
   size_t num_bytes = GRID_SIZE*sizeof(double3);
   CudaSafeCall(cudaMemcpy(d_V, V, num_bytes, cudaMemcpyHostToDevice));
   CudaSafeCall(cudaMemcpy(d_U, U, num_bytes, cudaMemcpyHostToDevice));
   CudaCheckError();
   cudaDeviceSynchronize();
}


