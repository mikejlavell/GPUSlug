#include <string>
#include <cuda_runtime.h>

#define ncores 8
//Simulation Parameters
const double gr_xbeg = 0.0;
const double gr_xend = 1.0;
const int sim_type = 1; // 2 3 4      1 = Sod, 2 = Rarefaction, 3 = Blast2, 4 = ShuOsher
 #define sim_gamma  1.4
 #define sim_riemann "hll" // "roe"; "hllc";
 #define sim_cfl 0.8
const double sim_tmax = 0.2;
// Sod problem
const double sim_shockLoc = 0.5;
const double sim_densL = 1.0;
const double sim_velxL = 0.0; // #-2. #0.
const double sim_presL = 1.0; // #0.4 #1.
const double sim_densR = 0.125; // #0.125
const double sim_velxR = 0.0;// #0.
const double sim_presR = 0.1;// #0.1
 

//-------------- Grid Parameters ---------------------------------
  __device__ const int gr_ngc = 2;
  __device__ const int N = 32768;
  __device__ const int GRID_SIZE = N + 2*gr_ngc;
  __device__ const double gr_dx = (gr_xend - gr_xbeg)/GRID_SIZE;
  __device__ const int gr_ibeg = 2; //0 is in the gaurd cells
  __device__ const int gr_iend = N + gr_ibeg - 1;
  __device__ const int gr_imax = GRID_SIZE -1;	//Because C/C++


// slope limiters
// MINMOD = 1
// VANLEER= 2
// MC     = 3
 __device__ int sim_limiter = 1;

// primitive vars
 __device__ int DENS_VAR =1;
 __device__ int VELX_VAR =2;
 __device__ int PRES_VAR =3;

// conservative vars
 __device__ int MOMX_VAR =2;
 __device__ int ENER_VAR =3;

// waves
 #define SHOCKLEFT 1
 #define CTENTROPY 2
 #define SHOCKRGHT 3
 __device__ const int NUMB_WAVE =3;

// BC
// __device__ int OUTFLOW  =1;
// __device__ int PERIODIC =2;
// __device__ int REFLECT  =3;
// __device__ int USER     =4;
 int sim_bcType = 1;

 #define pi  4.0*atan(1.0)
