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
__host__ __device__  void prim2cons(const float3 V,float3 &U)
    {
    float ekin, eint;

    U.x = V.x;
    U.y = V.x*V.y;

    ekin = 0.5f*V.x*V.y*V.y;
    eint = V.z/(sim_gamma-1.0f);
    U.z = ekin + eint;
}

/*----------------------- Initial Conditions ----------------------------------*/
void sim_init(float3 *gr_V, float3 *gr_U, float *gr_xCoord)
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
                   gr_V[i].x = 1.0f;
                   gr_V[i].y = 0.0f;
                   gr_V[i].z = 1000.0f;
               }
            else if (gr_xCoord[i] <= 0.9 && gr_xCoord[i] > 0.1)
              {
               //middle state
                   gr_V[i].x = 1.0f;
                   gr_V[i].y = 0.0f;
                   gr_V[i].z = 0.01f;
              }
            else
              {
                   gr_V[i].x = 1.0f;
                   gr_V[i].y = 0.0f;
                   gr_V[i].z = 100.0f;
             }
         }
      } 
    else if(sim_type == 4) // Shu-Osher
    {
        float x; 
        //IC for shu-osher problem
        //transform the domain [0,1] onto the one give for the problem:[-4.5,4.5]
#pragma omp parallel for num_threads(ncores) private(x)        
       for(int i = 0; i < GRID_SIZE; i++)
       {
            x = gr_xCoord[i];
            if (x < -4.0f)
            {
            //left state
                gr_V[i].x = 3.857143f;
                gr_V[i].y = 2.629369f;
                gr_V[i].z = 10.33333f;
            }   
       	    else
           {
           //Right State
                gr_V[i].x = 1.0f + 0.2f*sin(5.0f*x);
                gr_V[i].y = 0.0f;
                gr_V[i].z = 1.0f;
            }
        }
    }
#pragma omp parallel for num_threads(ncores)    
    for(int i = 0; i < GRID_SIZE; i++) prim2cons(gr_V[i], gr_U[i]);
}

//------------------------- CFL Condition ------------------------------------
float cfl(float3 *gr_V)
{   
    float cs, lambda, maxSpeed, dt; 
    maxSpeed = -1e30;
#pragma omp parallel for reduction(max:maxSpeed) num_threads(ncores) private(cs, lambda)
 for(int i = gr_ibeg; i < gr_iend; i++)
    {   
         cs = sqrtf(sim_gamma*gr_V[i].z/gr_V[i].x);
         lambda=(fabsf(gr_V[i].y) + cs);
         maxSpeed=fmaxf(maxSpeed,lambda);
     }
	//std::cout<< "max speed = "<< maxSpeed << std::endl;
  // cfl
  dt = sim_cfl*gr_dx/maxSpeed;
  return dt;
}
//------------------------- Boundary Condition Operators ---------------------
void bc_outflow(float3 *V)
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

void bc_reflect(float3 *V)
{
   for( int i = 0; i < gr_ngc; i++)
	{
       int k0 = gr_ngc;
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

void bc_periodic(float3 *V)
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

void bc_apply(float3 *U)
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

void grid_finalize( float *gr_xCoord, float3 *gr_U, float3 *gr_V, //Host
 float3 *d_gr_U, float3 *d_gr_V, float3 *d_gr_vL, float3 *d_gr_vR, float3 *d_gr_flux) //Device
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
void transfer_to_cpu(float3 *d_V, float3 *d_U, float3 *V, float3 *U) //gpu to cpu
{
   size_t num_bytes = GRID_SIZE*sizeof(float3);
   CudaSafeCall(cudaMemcpy(V, d_V, num_bytes, cudaMemcpyDeviceToHost));
   CudaSafeCall(cudaMemcpy(U, d_U, num_bytes, cudaMemcpyDeviceToHost));
   CudaCheckError();
   cudaDeviceSynchronize();
}

void transfer_to_gpu(float3 *V, float3 *U, float3 *d_V, float3 *d_U) //cpu to gpu
{
   size_t num_bytes = GRID_SIZE*sizeof(float3);
   CudaSafeCall(cudaMemcpy(d_V, V, num_bytes, cudaMemcpyHostToDevice));
   CudaSafeCall(cudaMemcpy(d_U, U, num_bytes, cudaMemcpyHostToDevice));
   CudaCheckError();
   cudaDeviceSynchronize();
}

__host__ __device__ void eigenvalues(float3 V,float lambda[NUMB_WAVE]){
    float  a, u;
    // sound speed
    a = sqrtf(sim_gamma*V.z/V.x);
    u = V.y;
    
    lambda[SHOCKLEFT] = u - a;
    lambda[CTENTROPY] = u;
    lambda[SHOCKRGHT] = u + a;
}


/*----------------------------------- EOS ---------------------------------------------- */
__host__ __device__ void eos_cell(const float dens,const float eint, float &pres)
{
       pres = fmaxf((sim_gamma-1.0f)*dens*eint,1e-5);
}

/*--------------------------------------------- Eigensystem -----------------------*/
  
__host__ __device__  void right_eigenvectors(float3 V,bool conservative,
 float3 reig[NUMB_WAVE]){
  //Right Eigenvectors

    float  a, u, d, g, ekin, hdai, hda;
    
    // sound speed, and others
    a = sqrtf(sim_gamma*V.z/V.x);
    u = V.y;
    d = V.x;
    g = sim_gamma - 1.0f;
    ekin = 0.5f*u*u;
    hdai = 0.5f*d/a;
    hda  = 0.5f*d*a;
    
    if (conservative == 1){
       //// Conservative eigenvector
       reig[SHOCKLEFT].x = 1.0f;
       reig[SHOCKLEFT].y = u - a;
       reig[SHOCKLEFT].z = ekin + a*a/g - a*u;
       reig[SHOCKLEFT].x *= -hdai;
       reig[SHOCKLEFT].y *= -hdai;
       reig[SHOCKLEFT].z *= -hdai;

       reig[CTENTROPY].x = 1.0f;
       reig[CTENTROPY].y = u;
       reig[CTENTROPY].z = ekin;
       
       reig[SHOCKRGHT].x = 1.0f;
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
       reig[SHOCKLEFT].y = 0.5f;
       reig[SHOCKLEFT].z = -hda;

       reig[CTENTROPY].x = 1.0f;
       reig[CTENTROPY].y = 0.0f;
       reig[CTENTROPY].z = 0.0f;

       reig[SHOCKRGHT].x = hdai;
       reig[SHOCKRGHT].y = 0.5f;
       reig[SHOCKRGHT].z = hda;         
    }   
}


__host__ __device__ void left_eigenvectors(float3 V,bool conservative,
 float3 leig[NUMB_WAVE]){ 
//Left Eigenvectors
    float  a, u, d, g, gi, ekin;
    
    // sound speed, and others
    a = sqrtf(sim_gamma*V.z/V.x);
    u = V.y;
    d = V.x;
    g = sim_gamma - 1.0f;
    gi = 1.0f/g;
    ekin = 0.5f*u*u;
 //   hdai = 0.5f*d/a;
 //   hda  = 0.5f*d*a;
    
    if (conservative == 1) {
       //// Conservative eigenvector
       leig[SHOCKLEFT].x = -ekin - a*u*gi;
       leig[SHOCKLEFT].y = u+a*gi;
       leig[SHOCKLEFT].z = -1.0f;
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
       leig[SHOCKRGHT].z = 1.0f;
       leig[SHOCKRGHT].x = g*leig[SHOCKRGHT].x/(d*a);
       leig[SHOCKRGHT].y = g*leig[SHOCKRGHT].y/(d*a);
       leig[SHOCKRGHT].z = g*leig[SHOCKRGHT].z/(d*a);

    }       
    else
    {
       //// Primitive eigenvector
       leig[SHOCKLEFT].x = 0.0f;
       leig[SHOCKLEFT].y = 1.0f;
       leig[SHOCKLEFT].z = -1.0f/(d*a);

       leig[CTENTROPY].x = 1.0f;
       leig[CTENTROPY].y = 0.0f;
       leig[CTENTROPY].z = -1.0f/(a*a);

       leig[SHOCKRGHT].x = 0.0f;
       leig[SHOCKRGHT].y = 1.0f;
       leig[SHOCKRGHT].z = 1.0f/(d*a);
    }
}




//GPU/host function to calculate the primitive variables given the conservative vars
__host__ __device__  void cons2prim(const float3 U,float3 &V)
    {
        float eint, ekin, pres;
        V.x = U.x;
        V.y = U.y/U.x;
        ekin = 0.5f*V.x*V.y*V.y;
        eint = fmaxf(U.z - ekin, 1e-5); //eint=rho*e
        eint = eint/U.x;
        // get pressure by calling eos
        eos_cell(U.x,eint,pres);
        V.z = pres;
    }    

//GPU/host function to calculate the analytic flux from a primitive variables
__host__ __device__  void prim2flux(const float3 V,float3 &Flux)
    {
    float ekin,eint,ener;
    Flux.x = V.x*V.y;
    Flux.y = Flux.x*V.y + V.z;
    ekin = 0.5f*V.y*V.y*V.x;
    eint = V.z/(sim_gamma-1.0f);
    ener = ekin + eint;
    Flux.z = V.y*(ener + V.z);
    }
    
//GPU/host function to calculate the analytic flux from a conservative variables
__host__ __device__  void cons2flux(const float3 U,float3 &Flux)
    {
    float3 V;
    cons2prim(U,V); //Transfer to Primitive
    prim2flux(V,Flux); //Calculate Flux
    }

/*----------------------------------------- Solution Update -------------------*/
__global__ void soln_update( float3 *U, float3 *V, const float3* Flux, const float dt)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x + gr_ngc -1;
    float dtx = dt/gr_dx;
    float3 temp = {0.0f, 0.0f, 0.0f};
    if(tid < GRID_SIZE -1) { 
    temp.x = U[tid].x - dtx*(Flux[tid+1].x - Flux[tid].x);
    temp.y = U[tid].y - dtx*(Flux[tid+1].y - Flux[tid].y);
    temp.z = U[tid].z - dtx*(Flux[tid+1].z - Flux[tid].z);
    __syncthreads();

    U[tid] = temp;
	__syncthreads();
    cons2prim(U[tid], V[tid]);
    }
}

 __device__ float dot_product(float3 u, float3 v){
    float ans = 0.0f;
    ans = u.x*v.x + u.y*v.y + u.z*v.z;
    return ans;
 }

__device__ void hll(const float3 vL, const float3 vR, float3 &Flux)
{
  float3 FL= {0.0f},FR= {0.0f},uL= {0.0f},uR= {0.0f};
  float aL,aR, sL, sR;

  //Fnding the Fluxes and Conservative Vars for HLL Flux.   
    prim2flux(vL,FL);
    prim2flux(vR,FR);
    prim2cons(vL,uL);
    prim2cons(vR,uR);
   
   //Sound Speeds 
    aL = sqrtf(sim_gamma*vL.z/vL.x);
    aR = sqrtf(sim_gamma*vR.z/vR.x);
    
    // fastest left and right going velocities
    sL = fminf(vL.y - aL, vR.y - aR);
    sR = fmaxf(vL.y + aL, vR.y + aR);
    
    if(sL >= 0.0f)  Flux = FL;   
    else if( sL < 0.0f && sR >= 0.0f) {
        Flux.x = (sR*FL.x - sL*FR.x + sR*sL*(uR.x - uL.x))/(sR - sL);   
        Flux.y = (sR*FL.y - sL*FR.y + sR*sL*(uR.y - uL.y))/(sR - sL);
        Flux.z = (sR*FL.z - sL*FR.z + sR*sL*(uR.z - uL.z))/(sR - sL);
    }
    else Flux = FR;
}


/*---------------------- ROE SOLVER --------------------------------*/
__device__ void roe(const float3 vL,const float3 vR, float3 &Flux)
{
  float3 FL= {0.0f},FR= {0.0f},uL= {0.0f},uR= {0.0f};
  float3 vAvg = {0.0f};
  float lambda[NUMB_WAVE] = {0.0f};
  float3 reig[NUMB_WAVE] = {0.0f}, leig[NUMB_WAVE] = {0.0f};
  int conservative = 1;
  float3 vec, sigma;
  
  // set the initial sum to be zero
  sigma = {0.0f};
  vec = {0.0f};

//Calculate the average state
	vAvg.x = 0.5f*(vL.x + vR.x);
	vAvg.y = 0.5f*(vL.y + vR.y);
	vAvg.z = 0.5f*(vL.z + vR.z);

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
     sigma.x += dot_product(leig[kWaveNum],vec)*fabsf(lambda[kWaveNum])*reig[kWaveNum].x;
     sigma.y += dot_product(leig[kWaveNum],vec)*fabsf(lambda[kWaveNum])*reig[kWaveNum].y;
     sigma.z += dot_product(leig[kWaveNum],vec)*fabsf(lambda[kWaveNum])*reig[kWaveNum].z;
   }
  __syncthreads();
  // numerical flux
  Flux.x = 0.5f*(FL.x + FR.x) - 0.5f*sigma.x; //Density
  Flux.y = 0.5f*(FL.y + FR.y) - 0.5f*sigma.y; //Momentum/Velocity
  Flux.z = 0.5f*(FL.z + FR.z) - 0.5f*sigma.z; //Energy/Pressure
}

/*------------------------------ Reconstruction Kernel ----------------------------*/
 __global__ void soln_reconstruct_PLM(const float dt, float3 *V, float3 *vl,
  float3 *vr)
 {
    int tid = threadIdx.x + blockIdx.x*blockDim.x + gr_ngc;
    if(tid < GRID_SIZE - gr_ngc)
    {
    float lambda[NUMB_WAVE] = {0.0f};
    float lambdaDtDx;
    float3 leig[NUMB_WAVE] = {0.0f};
    float3 reig[NUMB_WAVE] = {0.0f};
    float3 delL = {0.0f}, delR = {0.0f}; 
    float pL = 0.0f, pR = 0.0f, delW[NUMB_WAVE] = {0.0f};

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
       float3 sigL= {0.0f,0.0f,0.0f};
       float3 sigR= {0.0f,0.0f,0.0f};

     if (sim_riemann == "roe"){
	    for(int kWaveNum = 0; kWaveNum<NUMB_WAVE; kWaveNum++){
	      lambdaDtDx = lambda[kWaveNum]*dt/gr_dx;
              if (lambda[kWaveNum] > 0.0f){
              //Right Sum
             sigR.x += 0.5f*(1.0f - lambdaDtDx)*reig[kWaveNum].x*delW[kWaveNum];
	     	 sigR.y += 0.5f*(1.0f - lambdaDtDx)*reig[kWaveNum].y*delW[kWaveNum];
	     	 sigR.z += 0.5f*(1.0f - lambdaDtDx)*reig[kWaveNum].z*delW[kWaveNum];
				 __syncthreads();
		        }
              else if (lambda[kWaveNum] < 0.0f){
              //Left Sum
                  sigL.x += 0.5f*(-1.0f - lambdaDtDx)*reig[kWaveNum].x*delW[kWaveNum];
           		  sigL.y += 0.5f*(-1.0f - lambdaDtDx)*reig[kWaveNum].y*delW[kWaveNum];
         		  sigL.z += 0.5f*(-1.0f - lambdaDtDx)*reig[kWaveNum].z*delW[kWaveNum];
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
             sigR.x += 0.5f*(1.0f - lambdaDtDx)*reig[kWaveNum].x*delW[kWaveNum];
	     	 sigR.y += 0.5f*(1.0f - lambdaDtDx)*reig[kWaveNum].y*delW[kWaveNum];
	     	 sigR.z += 0.5f*(1.0f - lambdaDtDx)*reig[kWaveNum].z*delW[kWaveNum];
           //Left Sum
             sigL.x += 0.5f*(-1.0f - lambdaDtDx)*reig[kWaveNum].x*delW[kWaveNum];
             sigL.y += 0.5f*(-1.0f - lambdaDtDx)*reig[kWaveNum].y*delW[kWaveNum];
             sigL.z += 0.5f*(-1.0f - lambdaDtDx)*reig[kWaveNum].z*delW[kWaveNum];
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
__global__ void soln_getFlux(float dt, float3 *vl, float3 *vr, float3 *flux)
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
 void io_writeOutput(int ioCounter, float *x, float3 *U)
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
