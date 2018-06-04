#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>

#include "definition.h"

extern int GRID_SIZE;
extern int gr_dx;
extern int gr_dy;
extern int gr_ibeg;
extern int gr_iend;
extern int gr_jbeg;
extern int gr_jend;
extern int gr_imax;
extern int gr_jmax;
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
void grid_alloc( double *gr_xCoord, double *gr_yCoord,
 double4 *gr_U, double4 *gr_V, double4 *gr_vL, double4 *gr_vR, double4 *gr_flux, //Host
 double4 *d_gr_U, double4 *d_gr_V, double4 *d_gr_vL, double4 *d_gr_vR, double4 *d_gr_flux) //Device
{
 size_t num_bytes = GRID_XSZE*GRID_YSZE*sizeof(double4*);   
 //Allocating Host Variables
 gr_xCoord = (double*)malloc(GRID_XSZE*sizeof(double));
 gr_yCoord = (double*)malloc(GRID_YSZE*sizeof(double));
 gr_U = (double4*)malloc(num_bytes);   
 gr_V = (double4*)malloc(num_bytes);
 gr_vRX = (double4*)malloc(num_bytes);
 gr_vLX = (double4*)malloc(num_bytes);  
 gr_fluxX = (double4*)malloc(num_bytes);
 gr_vRY = (double4*)malloc(num_bytes);
 gr_vLY = (double4*)malloc(num_bytes);
 gr_fluxY = (double4*)malloc(num_bytes);

 //Allocating Device Variables
 CudaSafeCall(cudaMalloc((void**)&d_gr_U, num_bytes));
 CudaSafeCall(cudaMalloc((void**)&d_gr_V, num_bytes));
 CudaSafeCall(cudaMalloc((void**)&d_gr_vRX, num_bytes));
 CudaSafeCall(cudaMalloc((void**)&d_gr_vLX, num_bytes));
 CudaSafeCall(cudaMalloc((void**)&d_gr_fluxX, num_bytes));
 CudaSafeCall(cudaMalloc((void**)&d_gr_vRY, num_bytes));
 CudaSafeCall(cudaMalloc((void**)&d_gr_vLY, num_bytes));
 CudaSafeCall(cudaMalloc((void**)&d_gr_fluxY, num_bytes));

}


/*----------------------- Initial Conditions ----------------------------------*/
void sim_init(double4 *gr_V, double4 *gr_U, double *gr_xCoord, double gr_yCoord)
*/ Variable assignment
//     .x = dens
//     .y = velx
//     .z = vely
//     .w = pres
*/
{
    //initialize grid coordinates
#pragma omp parallel for num_threads(ncores)    
    for(int i = 0; i < GRID_XSZE; i ++ ) gr_xCoord[i] = i*gr_dx;
#pragma omp parallel for num_threads(ncores)    
    for(int j = 0; j < GRID_YSZE; j ++ ) gr_yCoord[j] = j*gr_dy;

    double small_number = 0.01;
    
    if(sim_type == 1) // Blast
    {
#pragma omp parallel for num_threads(ncores) 
	for(int i = 0; i < GRID_XSZE; i++ ){
		double x = gr_xCoord(i);
		for(int j = 0; j < GRID_YSZE; j++){
				double y = gr_yCoord(j);
				double r2 = x*x + y*y;

				gr_V.x[i][j] = 1.;
				gr_V.y[i][j] = 0.;
				gr_V.z[i][j] = 0.;

				if( r2 < 0.01 )
				       gr_V.w[i][j] = 10;
				else
					gr_V.w[i][j] = 0.1;
		}
	}

    }
    
    else if(sim_type == 2) // Kelvin-Helmoltz
    {
	// Need to find appropirate size for gr_ybeg, gr_xend
#pragma omp parallel for num_threads(ncores) 	
	for(int i = 0; i < GRID_XSZE; i++ ){
		double x = gr_xCoord(i);
		for(int j = 0; j < GRID_YSZE; j++){
			double y = gr_yCoord(j);

			double randx = rand() - 0.5;
			double randy = rand() - 0.5;

			if( abs(y) > 0.25 )
			{
				gr_V.x[i][j] = 1.0;
				gr_V.y[i][j] = -0.5;
			}
			else{
				gr_V.x[i][j] = 0.5;
				gr_V.y[i][j] = 2;
			}

			gr_V.y[i][j] = gr_V.y[i][j] + small*randx;
			gr_V.z[i][j] = small_number*randy;
			gr_V.w[i][j] = 2.5;

		}
	}
    }
    
/*    else if(sim_type == 3) // Richtmyer-Meshkov
    {

	// Simulation parameters    NEED TO SET THESE PARAMS
	sim_intPosn = 0.25;
	sim_perturb = 0.15915;
	sim_mixZone = 0.;
	sim_shockLoc = 1.0;

	sim_rhoL = 0.001351;
	sim_vxL = 0.0
	sim_vyL = 0.0
	sim_pL = 1.013e6;
	sim_gammaL = 1.1;

	sim_rhoR
	sim_vxR
	sim_vyR
	sim_pR
	sim_gammaR


#pragma omp parallel for num_threads(ncores) 
	for(int i = 0; i < GRID_XSZE; i++ ){
		double x = gr_xCoord(i);
		for(int j = 0; j < GRID_YSZE; j++){
			double y = gr_YSZE(j);
			double interTemp = sim_intPosn*gr_xmax;

			if (sim_pertub*cos(2.0*pi*y/gr_ymax) + intTemp > x)
			{
				gr_V[i][j].x = sim_rhoL;
				gr_V[i][j].y = sim_vxL;
				gr_V[i][j].z = sim_vyL;
				gr_V[i][j].w = sim_pL;  
			}
			else if(sim_perturb*cos(2.0*pi*y/gr_ymax) + intTemp \\
			     + sim_mixZone >= x && sim_perturb*cos(2.0*pi*y/gr_ymax) //
			     +intTemp <= x )
			{
				gr_V[i][j].x = 0.001351;
				gr_V[i][j].y = 0.;
				gr_V[i][j].z = 0.;
				gr_V[i][j].w = 1.013e6;  
			}
			

		}
	}
    }


    else if(sim_type == 4) // Richtmyer-Meshkov Samtany
    {
#pragma omp parallel for num_threads(ncores) 
	for(int i = 0; i < GRID_XSZE; i++ ){
		for(int j = 0; j < GRID_YSZE; j++){


		}
	}
    }
    
    else if(sim_type == 5) // Double Mach Reflection
    {
#pragma omp parallel for num_threads(ncores) 
	for(int i = 0; i < GRID_XSZE; i++ ){
		for(int j = 0; j < GRID_YSZE; j++){


		}
	}
    }

    else if(sim_type == 6) // Mach 3 Wind Tunnel
    {
#pragma omp parallel for num_threads(ncores) 
	for(int i = 0; i < GRID_XSZE; i++ ){
		for(int j = 0; j < GRID_YSZE; j++){


		}
	}
    }
*/

#pragma omp parallel for num_threads(ncores)    
    for(int i = 0; i < GRID_SIZE; i++) prim2cons(gr_V[i], gr_U[i]);
}


//------------------------- Boundary Condition Operators ---------------------

void bc_apply(double4 *U)
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

void bc_outflow(double4 *V)
{
#pragma omp parallel for num_threads(ncores) 
    for(int i = 0; i < gr_ngc ; i++)
    {
	for(int j = 0; j < gr_jmax; j++)
	{
      		// on the left bounday
      		V.x[i][j] = V.x[i+1][j];
      		V.y[i][j] = V.y[i+1][j];
      		V.z[i][j] = V.z[i+1][j]; 
      		V.w[i][j] = V.w[i+1][j];
       
		// on the right GC
		V.x[gr_imax+1-i][j] = V.x[gr_imax-i][j];
       		V.y[gr_imax+1-i][j] = V.y[gr_imax-i][j];
       		V.z[gr_imax+1-i][j] = V.z[gr_imax-i][j];              
		V.w[gr_imax+1-i][j] = V.w[gr_imax-1][j];
	}
    }
    for(int j = 0; j < gr_ngc; j++){
	for(int i = 0; i < gr_imax; i++){
		// on top bounday
      		V.x[i][j] = V.x[i][j+1];
      		V.y[i][j] = V.y[i][j+1];
      		V.z[i][j] = V.z[i][j+1];
      		V.w[i][j] = V.w[i][j+1];
		
		// on bottom bounday
		V.x[i][gr_jmax+1-1] = V.x[i][gr_jmax-i];
		V.y[i][gr_jmax+1-1] = V.y[i][gr_jmax-i];
		V.z[i][gr_jmax+1-1] = V.z[i][gr_jmax-i];
		V.w[i][gr_jmax+1-1] = V.w[i][gr_jmax-i];		
	}
    }
}

void bc_reflect(double3 *V)
{
#pragma omp parallel for num_threads(ncores) 	
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
#pragma omp parallel for num_threads(ncores) 
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
 double4 *gr_U, double4 *gr_V, double4 *gr_vLX, double4 *gr_vRX, double4 *gr_fluxX,
 double4 *gr_vLY, double4 *gr_vRY, double4 *gr_fluxY,                                      //Host
 double4 *d_gr_U, double4 *d_gr_V, double4 *d_gr_vLX, double4 *d_gr_vRX, double4 *d_gr_fluxX
 double4 *d_gr_vLY, double4 *d_gr_vRY, double4 *d_gr_fluxY)                                //Device
{
 //Deallocating Host Variables
 free(gr_xCoord);
 free(gr_U);
 free(gr_V);
 free(gr_vRX);
 free(gr_vLX);
 free(gr_fluxX);
 free(gr_vRY)
 free(gr_vLY);
 free(gr_fluxY);
 
 //Deallocating Device Variables
 cudaFree(d_gr_U);
 cudaFree(d_gr_V);
 cudaFree(d_gr_vRX);
 cudaFree(d_gr_vLX);
 cudaFree(d_gr_fluxX);
 cudaFree(d_gr_vRY);
 cudaFree(d_gr_vLY);
 cudaFree(d_gr_fluxY);
}


/*----------------- Transfer Routines ------------------------------------- */ 
void transfer_to_cpu(double4 *d_V, double4 *d_U, double4 *V, double4 *U) //gpu to cpu
{
   size_t num_bytes = GRID_XSZE*GRID_YSZE*sizeof(double4);
   CudaSafeCall(cudaMemcpy(V, d_V, num_bytes, cudaMemcpyDeviceToHost));
   CudaSafeCall(cudaMemcpy(U, d_U, num_bytes, cudaMemcpyDeviceToHost));
   CudaCheckError();
   cudaDeviceSynchronize();
}

void transfer_to_gpu(double4 *V, double4 *U, double4 *d_V, double4 *d_U) //cpu to gpu
{
   size_t num_bytes = GRID_XSZE*GRID_YSZE*sizeof(double4);
   CudaSafeCall(cudaMemcpy(d_V, V, num_bytes, cudaMemcpyHostToDevice));
   CudaSafeCall(cudaMemcpy(d_U, U, num_bytes, cudaMemcpyHostToDevice));
   CudaCheckError();
   cudaDeviceSynchronize();
}

