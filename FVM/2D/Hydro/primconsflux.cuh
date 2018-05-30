/* This header file contains the functions to Switch Variable Type and Calculate Flux*/


#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>
#include <algorithm>

#include "definition.h"
#include "eos.cuh"

//GPU/host function to calculate the conservative variables given the primitive vars
__host__ __device__  void prim2cons(const double4 V, double4 &U)
    {
    double eknx, ekny, eint;

    U.x = V.x;
    U.y = V.x*V.y;
    U.z = V.x*V.z;

    eknx = 0.5*V.x*V.y*V.y;
    ekny = 0.5*V.x*V.z*V.z;
    
    eint = V.p/(sim_gamma-1.0);
    U.w = eknx + ekny + eint;
}

//GPU/host function to calculate the primitive variables given the conservative vars
__host__ __device__  void cons2prim(const double4 U,double4 &V)
    {
        double eint, eknx, ekny, pres;

        V.x = U.x;
        V.y = U.y/U.x;
	V.z = U.z/U.x;

        eknx = 0.5*V.x*V.y*V.y;
	ekny = 0.5*V.x*V.z*V.z;
        eint = max(U.w - ekin, sim_smallPres); //eint=rho*e
        eint = eint/U.x;

        // get pressure by calling eos
        eos_cell(U.x,eint,sim_gamma,pres);
        V.w = pres;
    }    

//GPU/host function to calculate the analytic flux from a primitive variables
__host__ __device__  void prim2flux(const double4 V,double4 &Flux)
    {
    double eknx,ekny,eint,ener;
    Flux.x = V.x*V.y;
    Flux.y = V.x*V.z;
    Flux.z = Flux.x*V.y + Flux.y*V.z + V.w;

    eknx = 0.5*V.y*V.y*V.x;
    ekny = 0.5*V.z*V.z*V.x;
    eint = V.w/(sim_gamma-1.0);
    ener = eknx + ekny + eint;

    Flux.w = V.y*(ener + V.w);     // check this line!!!!!!
    }
    
//GPU/host function to calculate the analytic flux from a conservative variables
__host__ __device__  void prim2flux(const double4 U, double4 &Flux)
    {
    double4 V;
    cons2prim(U,V); //Transfer to Primitive
    prim2flux(V,Flux); //Calculate Flux
    }

