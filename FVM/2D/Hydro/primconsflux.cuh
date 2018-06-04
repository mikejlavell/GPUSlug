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
    
    eint = V.w/(sim_gamma-1.0);
    U.w = eknx + ekny + eint;
}

//GPU/host function to calculate the primitive variables given the conservative vars
__host__ __device__  void cons2prim(const double4 U,double4 &V)
    {
        double eint, eknx, ekny, pres;

        V.x = U.x;
        V.y = U.y/U.x;
	V.z = U.z/U.x;

        ekin = 0.5*V.x*(V.y*V.y + V.z*V.z);
        eint = max(U.w - ekin, sim_smallPres); //eint=rho*e
        eint = eint/U.x;

        // get pressure by calling eos
        eos_cell(U.x,eint,pres,sim_gamma);
        V.w = pres;
    }    

//GPU/host function to calculate the analytic flux from a primitive variables
__host__ __device__  void prim2flux(const double4 V,double4 &Flux, const int dir)
    {
    if( dir==0 ){
	double u1=V.y;
	double u2=V.z;}
    else if( dir==1 ){
	double u1=V.z; 
	double u2=V.y;}

    double eknx,ekny,eint,ener;
    ekin = 0.5(u1*u1 + u2*u2);
    eint = V.w/(sim_gamma-1.0);

    Flux.x = V.x*u1;
    if( dir==0 ){
	Flux.y = Flux.x*u1 + V.w;
	Flux.z = Flux.x*u2;}
    if( dir==1 ){
	Flux.z = Flux.x*u1 + V.w;
	Flux.y = Flux.x*u2;}
    Flux.w = u1*(ekin+eint+V.w);
    }

    
//GPU/host function to calculate the analytic flux from a conservative variables
__host__ __device__  void cons2flux(const double4 U, double4 &Flux)
    {
    double4 V;
    cons2prim(U,V); //Transfer to Primitive
    prim2flux(V,Flux); //Calculate Flux
    }

