#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>

//#ifndef defSlughelper
//#define defSlughelper

#include "definition.h"
#include "primconsflux.cuh"


/*----------------------CUDA Error stuff -------------------------------- */ 
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line);
inline void __cudaCheckError(const char *file, const int line);

/*--------------------- Allocate Variables ----------------------------------*/
void grid_alloc( double *&gr_xCoord, double *&gr_yCoord, double4 *&gr_U, double4 *&gr_V, 
 double4 *&d_gr_U, double4 *&d_gr_V, double4 *&d_gr_vLX, double4 *&d_gr_vRX, double4 *&d_gr_fluxX,
 double4 *&d_gr_vLY, double4 *&d_gr_vRY, double4 *&d_gr_fluxY); //Device

/*----------------------- Initial Conditions ----------------------------------*/
void sim_init(double4 *gr_V, double4 *gr_U, double *gr_xCoord, double *gr_yCoord);

//------------------------- Boundary Condition Operators ---------------------

void bc_apply(double4 *U);


void bc_outflow(double4 *V);

/*----------------- Finalize the grid ---------------------------------------*/

void grid_finalize( double *&gr_xCoord, double *&gr_yCoord, double4 *&gr_U, double4 *&gr_V,  //Host
 double4 *&d_gr_U, double4 *&d_gr_V, double4 *&d_gr_vLX, double4 *&d_gr_vRX, double4 *&d_gr_fluxX,
 double4 *&d_gr_vLY, double4 *&d_gr_vRY, double4 *&d_gr_fluxY);                              //Device

/*----------------- Transfer Routines ------------------------------------- */ 
void transfer_to_cpu(double4 *d_V, double4 *d_U, double4 *V, double4 *U); //gpu to cpu
void transfer_to_gpu(double4 *V, double4 *U, double4 *d_V, double4 *d_U); //cpu to gpu

void soln_ReconEvolveAvg(double dt, double4 *d_gr_V, double4 *d_gr_U,
                       double4 * d_gr_vlx, double4 *d_gr_vrx, double4 *d_gr_fluxx,
                       double4 * d_gr_vly, double4 *d_gr_vry, double4 *d_gr_fluxy, 
                      dim3 grid, dim3 block);

__global__ void soln_update(double4 *U, double4 *V, const double4 *Flux,
                            const double4 *Fluy, const double dt);


//#endif
