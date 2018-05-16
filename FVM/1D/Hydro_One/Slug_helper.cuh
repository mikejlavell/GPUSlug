#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>

#include "definition.h"
#include "slope_limiter.cuh"


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



//GPU/host function to calculate the conservative variables given the primitive vars
__host__ __device__  void prim2cons(const double3 V,double3 &U)
    {
    double ekin, eint;

    U.x = V.x;
    U.y = V.x*V.y;

    ekin = 0.5*V.x*V.y*V.y;
    eint = V.z/(sim_gamma-1.0);
    U.z = ekin + eint;
}

/*----------------------- Initial Conditions ----------------------------------*/
void sim_init(double3 *gr_V, double3 *gr_U, double *gr_xCoord)
{
    //initialize grid coordinates
#pragma omp parallel for num_threads(ncores)    
    for(int i = 0; i < GRID_SIZE; i ++ ) gr_xCoord[i] = gr_xbeg +  i*gr_dx;

    if(sim_type == 1) // Sod's Shock Tube
     {
        for(int i = 0; i < GRID_SIZE; i++)
            {
                if (gr_xCoord[i] < sim_shockLoc)
                {
                   gr_V[i].x = sim_densL;
                   gr_V[i].y = sim_velxL;
                   gr_V[i].z = sim_presL;
		}
                else
                {
                   gr_V[i].x = sim_densR;
                   gr_V[i].y = sim_velxR;
                   gr_V[i].z = sim_presR;
                }   
            }
      }
           
    else if(sim_type == 2) // Rarefaction
    {
    }
    else if(sim_type == 3) // Blast2
    {
#pragma omp parallel for num_threads(ncores)    
        for(int i = 0; i<GRID_SIZE; i++) 
        {
            if (gr_xCoord[i] <= 0.1)
               {
                   gr_V[i].x = 1.0;
                   gr_V[i].y = 0.0;
                   gr_V[i].z = 1000.0;
               }
            else if (gr_xCoord[i] <= 0.9 && gr_xCoord[i] > 0.1)
              {
               //middle state
                   gr_V[i].x = 1.0;
                   gr_V[i].y = 0.0;
                   gr_V[i].z = 0.01;
              }
            else
              {
                   gr_V[i].x = 1.0;
                   gr_V[i].y = 0.0;
                   gr_V[i].z = 100.0;
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
            x = gr_xCoord[i];
            if (x < -4.0)
            {
            //left state
                gr_V[i].x = 3.857143;
                gr_V[i].y = 2.629369;
                gr_V[i].z = 10.33333;
            }   
       	    else
           {
           //Right State
                gr_V[i].x = 1 + 0.2*sin(5.0*x);
                gr_V[i].y = 0.0;
                gr_V[i].z = 1.0;
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
 for(int i = gr_ibeg; i < gr_iend; i++)
    {   
         cs = sqrtf(sim_gamma*gr_V[i].z/gr_V[i].x);
         lambda=(fabs(gr_V[i].y) + cs);
         maxSpeed=fmax(maxSpeed,lambda);
     }
	//std::cout<< "max speed = "<< maxSpeed << std::endl;
  // cfl
  dt = sim_cfl*gr_dx/maxSpeed;
  return dt;
}
//------------------------- Boundary Condition Operators ---------------------
void bc_outflow(double3 *V)
{
    for(int i = gr_ngc; i > 0 ; i--)
    {
       // on the left GC
       V[i].x = V[i+1].x;
       V[i].y = V[i+1].y;
       V[i].z = V[i+1].z;       
       // on the right GC
       V[gr_imax-i].x = V[gr_imax-1-i].x;
       V[gr_imax-i].y = V[gr_imax-1-i].y;
       V[gr_imax-i].z = V[gr_imax-1-i].z;              
    }
}

void bc_reflect(double3 *V)
{
   for( int i = 0; i < gr_ngc; i++)
	{
       int k0 = 2*gr_ngc;
       int k1 = gr_iend-gr_ngc;

       // on the left GC
       V[i].x = V[k0-i].x;
       V[i].y =-V[k0-i].y;
       V[i].z = V[k0-i].z;

       // on the right GC
       V[k1+k0-i].x = V[k1+i].x;
       V[k1+k0-i].y =-V[k1+i].y;
       V[k1+k0-i].z = V[k1+i].z;
	}
}

void bc_periodic(double3 *V)
{
     for(int i = 0; i < gr_ngc; i++)
     {  
       V[gr_iend+i].x = V[ gr_ngc + i].x;
       V[gr_iend+i].y = V[ gr_ngc + i].y;
       V[gr_iend+i].z = V[ gr_ngc + i].z;
       
       V[gr_ibeg-i].x = V[gr_iend - i+1].x;
       V[gr_ibeg-i].y = V[gr_iend - i+1].y;
       V[gr_ibeg-i].z = V[gr_iend - i+1].z;       
     }
}

void bc_apply(double3 *U)
{
    if (sim_bcType == 1)
    { // Outflow
       bc_outflow(U);
    }
    else if (sim_bcType == 4) //User 
    {
        //Do nothing, it's for the Shu-Osher Problem
    }   
    else if (sim_bcType == 3) //Reflect
    {
       bc_reflect(U);
    }   
    else if (sim_bcType == 2) //Periodic
    {
       bc_periodic(U);
    }
}



/*----------------- Finalize the grid ---------------------------------------*/

void grid_finalize( double *gr_xCoord, double3 *gr_U, double3 *gr_V, //Host
 double3 *d_gr_U, double3 *d_gr_V, double3 *d_gr_vL, double3 *d_gr_vR, double3 *d_gr_flux) //Device
{
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

__host__ __device__ void eigenvalues(double3 V,double lambda[NUMB_WAVE]){
    double  a, u;
    // sound speed
    a = sqrtf(sim_gamma*V.z/V.x);
    u = V.y;
    
    lambda[SHOCKLEFT] = u - a;
    lambda[CTENTROPY] = u;
    lambda[SHOCKRGHT] = u + a;
}


/*----------------------------------- EOS ---------------------------------------------- */
__host__ __device__ void eos_cell(const double dens,const double eint, double &pres)
{
       pres = fmax((sim_gamma-1.)*dens*eint,1e-6);
}

/*--------------------------------------------- Eigensystem -----------------------*/
  
__host__ __device__  void right_eigenvectors(double3 V,bool conservative,
 double3 reig[NUMB_WAVE]){
  //Right Eigenvectors

    double  a, u, d, g, ekin, hdai, hda;
    
    // sound speed, and others
    a = sqrtf(sim_gamma*V.z/V.x);
    u = V.y;
    d = V.x;
    g = sim_gamma - 1.0;
    ekin = 0.5*u*u;
    hdai = 0.5*d/a;
    hda  = 0.5*d*a;
    
    if (conservative == 1){
       //// Conservative eigenvector
       reig[SHOCKLEFT].x = 1.0;
       reig[SHOCKLEFT].y = u - a;
       reig[SHOCKLEFT].z = ekin + a*a/g - a*u;
       reig[SHOCKLEFT].x *= -hdai;
       reig[SHOCKLEFT].y *= -hdai;
       reig[SHOCKLEFT].z *= -hdai;

       reig[CTENTROPY].x = 1.0;
       reig[CTENTROPY].y = u;
       reig[CTENTROPY].z = ekin;
       
       reig[SHOCKRGHT].x = 1.0;
       reig[SHOCKRGHT].y = u + a;
       reig[SHOCKRGHT].z = ekin + a*a/g + a*u;
       reig[SHOCKRGHT].x *= hdai;
       reig[SHOCKRGHT].y *= hdai;
       reig[SHOCKRGHT].z *= hdai;
       }
    else
    {
       //// Primitive eigenvector
       reig[SHOCKLEFT].x = -hdai;
       reig[SHOCKLEFT].y = 0.5;
       reig[SHOCKLEFT].z = -hda;

       reig[CTENTROPY].x = 1.0;
       reig[CTENTROPY].y = 0.0;
       reig[CTENTROPY].z = 0.0;

       reig[SHOCKRGHT].x = hdai;
       reig[SHOCKRGHT].y = 0.5;
       reig[SHOCKRGHT].z = hda;         
    }   
}


__host__ __device__ void left_eigenvectors(double3 V,bool conservative,
 double3 leig[NUMB_WAVE]){ 
//Left Eigenvectors
    double  a, u, d, g, gi, ekin;
    
    // sound speed, and others
    a = sqrtf(sim_gamma*V.z/V.x);
    u = V.y;
    d = V.x;
    g = sim_gamma - 1.0;
    gi = 1.0/g;
    ekin = 0.5*u*u;
 //   hdai = 0.5*d/a;
 //   hda  = 0.5*d*a;
    
    if (conservative == 1) {
       //// Conservative eigenvector
       leig[SHOCKLEFT].x = -ekin - a*u*gi;
       leig[SHOCKLEFT].y = u+a*gi;
       leig[SHOCKLEFT].z = -1.0;
       leig[SHOCKLEFT].x = g*leig[SHOCKLEFT].x/(d*a);
       leig[SHOCKLEFT].y = g*leig[SHOCKLEFT].y/(d*a);
       leig[SHOCKLEFT].z = g*leig[SHOCKLEFT].z/(d*a);

       leig[CTENTROPY].x = d*(-ekin + gi*a*a)/a;
       leig[CTENTROPY].y = d*u/a;
       leig[CTENTROPY].z = -d/a;
       leig[CTENTROPY].x = g*leig[CTENTROPY].x/(d*a);
       leig[CTENTROPY].y = g*leig[CTENTROPY].y/(d*a);
       leig[CTENTROPY].z = g*leig[CTENTROPY].z/(d*a);
       
       leig[SHOCKRGHT].x = ekin - a*u*gi;
       leig[SHOCKRGHT].y = -u+a*gi;
       leig[SHOCKRGHT].z = 1.0;
       leig[SHOCKRGHT].x = g*leig[SHOCKRGHT].x/(d*a);
       leig[SHOCKRGHT].y = g*leig[SHOCKRGHT].y/(d*a);
       leig[SHOCKRGHT].z = g*leig[SHOCKRGHT].z/(d*a);

    }       
    else
    {
       //// Primitive eigenvector
       leig[SHOCKLEFT].x = 0.0;
       leig[SHOCKLEFT].y = 1.0;
       leig[SHOCKLEFT].z = -1.0/(d*a);

       leig[CTENTROPY].x = 1.0;
       leig[CTENTROPY].y = 0.0;
       leig[CTENTROPY].z = -1.0/(a*a);

       leig[SHOCKRGHT].x = 0.0;
       leig[SHOCKRGHT].y = 1.0;
       leig[SHOCKRGHT].z = 1.0/(d*a);
    }
}




//GPU/host function to calculate the primitive variables given the conservative vars
__host__ __device__  void cons2prim(const double3 U,double3 &V)
    {
        double eint, ekin, pres;
        V.x = U.x;
        V.y = U.y/U.x;
        ekin = 0.5*V.x*V.y*V.y;
        eint = max(U.z - ekin, 1e-6); //eint=rho*e
        eint = eint/U.x;
        // get pressure by calling eos
        eos_cell(U.x,eint,pres);
        V.z = pres;
    }    

//GPU/host function to calculate the analytic flux from a primitive variables
__host__ __device__  void prim2flux(const double3 V,double3 &Flux)
    {
    double ekin,eint,ener;
    Flux.x = V.x*V.y;
    Flux.y = Flux.x*V.y + V.z;
    ekin = 0.5*V.y*V.y*V.x;
    eint = V.z/(sim_gamma-1.0);
    ener = ekin + eint;
    Flux.z = V.y*(ener + V.z);
    }
    
//GPU/host function to calculate the analytic flux from a conservative variables
__host__ __device__  void cons2flux(const double3 U,double3 &Flux)
    {
    double3 V;
    cons2prim(U,V); //Transfer to Primitive
    prim2flux(V,Flux); //Calculate Flux
    }

/*----------------------------------------- Solution Update -------------------*/
__global__ void soln_update( double3 *U, double3 *V, const double3* Flux, const double dt)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x + gr_ngc;
    double dtx = dt/gr_dx;
    double3 temp = {0.0, 0.0, 0.0};
    if(tid < GRID_SIZE - gr_ngc) { 
    temp.x = U[tid].x - dtx*(Flux[tid+1].x - Flux[tid].x);
    temp.y = U[tid].y - dtx*(Flux[tid+1].y - Flux[tid].y);
    temp.z = U[tid].z - dtx*(Flux[tid+1].z - Flux[tid].z);
    __syncthreads();

    U[tid] = temp;
	__syncthreads();
    cons2prim(U[tid], V[tid]);
    }
}

 __device__ double dot_product(double3 u, double3 v){
    double ans = 0.0;
    ans = u.x*v.x + u.y*v.y + u.z*v.z;
    return ans;
 }

__device__ void hll(const double3 vL, const double3 vR, double3 &Flux)
{
  double3 FL= {0.0},FR= {0.0},uL= {0.0},uR= {0.0};
  double aL,aR, sL, sR;

  //Fnding the Fluxes and Conservative Vars for HLL Flux.   
    prim2flux(vL,FL);
    prim2flux(vR,FR);
    prim2cons(vL,uL);
    prim2cons(vR,uR);
   
   //Sound Speeds 
    aL = sqrtf(sim_gamma*vL.z/vL.x);
    aR = sqrtf(sim_gamma*vR.z/vR.x);
    
    // fastest left and right going velocities
    sL = fmin(vL.y - aL, vR.y - aR);
    sR = fmax(vL.y + aL, vR.y + aR);
    
    if(sL >= 0.0)  Flux = FL;   
    else if( sL < 0.0 && sR >= 0.0) {
        Flux.x = (sR*FL.x - sL*FR.x + sR*sL*(uR.x - uL.x))/(sR - sL);   
        Flux.y = (sR*FL.y - sL*FR.y + sR*sL*(uR.y - uL.y))/(sR - sL);
        Flux.z = (sR*FL.z - sL*FR.z + sR*sL*(uR.z - uL.z))/(sR - sL);
    }
    else Flux = FR;
}


/*---------------------- ROE SOLVER --------------------------------*/
__device__ void roe(const double3 vL,const double3 vR, double3 &Flux)
{
  double3 FL= {0.0},FR= {0.0},uL= {0.0},uR= {0.0};
  double3 vAvg = {0.0};
  double lambda[NUMB_WAVE] = {0.0};
  double3 reig[NUMB_WAVE] = {0.0}, leig[NUMB_WAVE] = {0.0};
  int conservative = 1;
  double3 vec, sigma;
  
  // set the initial sum to be zero
  sigma = {0.0};
  vec = {0.0};

//Calculate the average state
	vAvg.x = 0.5*(vL.x + vR.x);
	vAvg.y = 0.5*(vL.y + vR.y);
	vAvg.z = 0.5*(vL.z + vR.z);

//Find Average States and Eigenvalues + Eigenvectors for Averaged State  
    eigenvalues(vAvg,lambda);
    left_eigenvectors(vAvg,conservative,leig);
    right_eigenvectors(vAvg,conservative,reig);
    
//Fnding the Fluxes and Conservative Vars for Roe Flux.   
    prim2flux(vL,FL);
    prim2flux(vR,FR);
    prim2cons(vL,uL);
    prim2cons(vR,uR);

    vec.x = uR.x - uL.x;
    vec.y = uR.y - uL.y;
    vec.z = uR.z - uL.z;
	
  for(int kWaveNum = 0;kWaveNum<NUMB_WAVE;kWaveNum++)
  {
     sigma.x += dot_product(leig[kWaveNum],vec)*fabs(lambda[kWaveNum])*reig[kWaveNum].x;
     sigma.y += dot_product(leig[kWaveNum],vec)*fabs(lambda[kWaveNum])*reig[kWaveNum].y;
     sigma.z += dot_product(leig[kWaveNum],vec)*fabs(lambda[kWaveNum])*reig[kWaveNum].z;
   }
  __syncthreads();
  // numerical flux
  Flux.x = 0.5*(FL.x + FR.x) - 0.5*sigma.x; //Density
  Flux.y = 0.5*(FL.y + FR.y) - 0.5*sigma.y; //Momentum/Velocity
  Flux.z = 0.5*(FL.z + FR.z) - 0.5*sigma.z; //Energy/Pressure
}

/*------------------------------ Reconstruction Kernel ----------------------------*/
 __global__ void soln_reconstruct_PLM(const double dt, double3 *V, double3 *vl,
  double3 *vr)
 {
    int tid = threadIdx.x + blockIdx.x*blockDim.x + gr_ngc;
    if(tid < GRID_SIZE - gr_ngc)
    {
    double lambda[NUMB_WAVE] = {0.0};
    double lambdaDtDx;
    double3 leig[NUMB_WAVE] = {0.0};
    double3 reig[NUMB_WAVE] = {0.0};
    double3 delL = {0.0}, delR = {0.0}; 
    double pL = 0.0, pR = 0.0, delW[NUMB_WAVE] = {0.0};

    eigenvalues(V[tid],lambda); //Get Eigenvalues for reconstruction
    left_eigenvectors(V[tid],0,leig);
    right_eigenvectors(V[tid],0,reig);
	__syncthreads();

        for(int kWaveNum = 0; kWaveNum < NUMB_WAVE; kWaveNum ++)
        {
           delL.x = V[tid].x - V[tid-1].x;
           delL.y = V[tid].y - V[tid-1].y;
           delL.z = V[tid].z - V[tid-1].z;
           delR.x = V[tid+1].x - V[tid].x;
           delR.y = V[tid+1].y - V[tid].y;
           delR.z = V[tid+1].z - V[tid].z;
		   __syncthreads();
           // project onto characteristic vars
           pL = dot_product(leig[kWaveNum], delL);
           pR = dot_product(leig[kWaveNum], delR);
           // Use a TVD Slope limiter
           if (sim_limiter == 1){
                delW[kWaveNum] = minmod(pL, pR);
              }
           else if (sim_limiter == 2){
                delW[kWaveNum] = vanLeer(pL, pR);
              }
           else if (sim_limiter == 3){
                delW[kWaveNum] = mc(pL, pR);
              }
			 __syncthreads();
        }   
       
         //do char tracing
        //set the initial sum to be zero
       double3 sigL= {0.0,0.0,0.0};
       double3 sigR= {0.0,0.0,0.0};

     if (sim_riemann == "roe"){
	    for(int kWaveNum = 0; kWaveNum<NUMB_WAVE; kWaveNum++){
	      lambdaDtDx = lambda[kWaveNum]*dt/gr_dx;
              if (lambda[kWaveNum] > 0.0){
              //Right Sum
             sigR.x += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].x*delW[kWaveNum];
	     	 sigR.y += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].y*delW[kWaveNum];
	     	 sigR.z += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].z*delW[kWaveNum];
				 __syncthreads();
		        }
              else if (lambda[kWaveNum] < 0.0){
              //Left Sum
                  sigL.x += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].x*delW[kWaveNum];
           		  sigL.y += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].y*delW[kWaveNum];
         		  sigL.z += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].z*delW[kWaveNum];
                 __syncthreads();
        		}
	        }
         }
      else if (sim_riemann == "hll" || sim_riemann == "hllc")
	     {
	       for(int kWaveNum = 0; kWaveNum<NUMB_WAVE; kWaveNum++)
		    {
		     lambdaDtDx = lambda[kWaveNum]*dt/gr_dx;
            //Right Sum
             sigR.x += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].x*delW[kWaveNum];
	     	 sigR.y += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].y*delW[kWaveNum];
	     	 sigR.z += 0.5*(1.0 - lambdaDtDx)*reig[kWaveNum].z*delW[kWaveNum];
           //Left Sum
             sigL.x += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].x*delW[kWaveNum];
             sigL.y += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].y*delW[kWaveNum];
             sigL.z += 0.5*(-1.0 - lambdaDtDx)*reig[kWaveNum].z*delW[kWaveNum];
             __syncthreads();
	    	}     	
	    }

           // Now PLM reconstruction for dens, velx, and pres
           vl[tid].x = V[tid].x + sigL.x;
           vl[tid].y = V[tid].y + sigL.y;
           vl[tid].z = V[tid].z + sigL.z;
           vr[tid].x = V[tid].x + sigR.x;
           vr[tid].y = V[tid].y + sigR.y;
           vr[tid].z = V[tid].z + sigR.z;   
     }
 }

/*---------------------------------- Get Flux Kernel ---------------------------*/ 
__global__ void soln_getFlux(double dt, double3 *vl, double3 *vr, double3 *flux)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x + gr_ngc;
	if(tid < GRID_SIZE - gr_ngc) 
	{
		if(sim_riemann == "hll")
		{
			hll(vr[tid-1],vl[tid],flux[tid]);//Call the HLL Riemann Solver
		}
/*		if(sim_riemann == "hllc")
		{
			hllc(vl[tid],vr[tid-1],flux[tid]);//Call the HLLC Riemann Solver
		}//*/
		if(sim_riemann == "roe")
		{
			roe(vr[tid-1],vl[tid],flux[tid]);//Call the Roe Riemann Solver
		}
	}
}

/*--------------------- Function to export data to .dat file ----------------*/
 void io_writeOutput(int ioCounter, double *x, double3 *U)
 {  
    std::string sim;        
    std::ofstream myfile;
    if (sim_type == 1) sim = "Sod";
    else if(sim_type == 2) sim = "Rare";
    else if(sim_type == 3) sim = "Blast2";
    else if(sim_type == 4) sim = "ShuOsher";
    myfile.open("slug"+std::to_string(ioCounter)+sim+".dat");
    
    for(int j = 0; j<GRID_SIZE; j++){
        myfile << x[j] << '\t';
    }
    myfile << std::endl;

    for(int j = 0; j<GRID_SIZE; j++){
        myfile << U[j].x << '\t';
    }
    myfile << std::endl;
    for(int j = 0; j<GRID_SIZE; j++){
        myfile << U[j].y << '\t';
    }
    myfile << std::endl;
    for(int j = 0; j<GRID_SIZE; j++){
        myfile << U[j].z << '\t';
    }
    myfile << std::endl;
    myfile.close();
 }
