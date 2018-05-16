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
__host__ __device__  void prim2cons(const double3 V,double3 &U)
    {
    double ekin, eint;

    U.x = V.x;
    U.y = V.x*V.y;

    ekin = 0.5*V.x*V.y*V.y;
    eint = V.z/(sim_gamma-1.0);
    U.z = ekin + eint;
}

//GPU/host function to calculate the primitive variables given the conservative vars
__host__ __device__  void cons2prim(const double3 U,double3 &V)
    {
        double eint, ekin, pres;
        V.x = U.x;
        V.y = U.y/U.x;
        ekin = 0.5*V.x*V.y*V.y;
        eint = max(U.z - ekin, sim_smallPres); //eint=rho*e
        eint = eint/U.x;
        // get pressure by calling eos
        eos_cell(U.x,eint,sim_gamma,pres);
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
__host__ __device__  void prim2flux(const double3 U,double3 &Flux)
    {
    double3 V;
    cons2prim(U,V); //Transfer to Primitive
    prim2flux(V,Flux); //Calculate Flux
    }

