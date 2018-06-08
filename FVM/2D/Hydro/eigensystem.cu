#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>

#include "definition.h"

/*--------------- Eigenvalues --------------------*/
__host__ __device__ void eigenvalues(double4 V,double lambda[NUMB_WAVE], int dir){
    
    double  a, u;

    if (dir == 0)
	u = V.y;
    else if(dir == 1)
	u = V.z;

    // sound speed
    a = sqrtf(sim_gamma*V.w/V.x);
    
    lambda[SHOCKLEFT] = u - a;
    lambda[SLOWWLEFT] = u;
    lambda[CTENTROPY] = u;
    lambda[SHOCKRGHT] = u + a;
}



/*--------------- Right Eigenvectors --------------------*/ 
__host__ __device__ void right_eigenvectors(double4 V, bool conservative, 
			                    double4 reig[NUMB_WAVE], int dir){

    double  a, u1, u2, d, g, ekin, hdai, hda;

    // velocity
    if (dir == 0){
	u1 = V.y;
	u2 = V.z;}
    else if(dir == 1){
	u1 = V.z;
	u2 = V.y;}
    
    // sound speed, and others
    a = sqrt(sim_gamma*V.w/V.x);
    d = V.x;
    g = sim_gamma - 1.0;
    ekin = 0.5*(u1*u1 + u2*u2);
    hdai = 0.5*d/a;
    
    if (conservative==1){
       //// Conservative eigenvector
       if( dir==0 ){
	  reig[SHOCKLEFT].y = u1-a;
	  reig[SHOCKLEFT].z = u2;
	  
	  reig[SLOWWLEFT].y = 0.0;
	  reig[SLOWWLEFT].z = 1.0;

	  reig[CTENTROPY].y = u1;
	  reig[CTENTROPY].z = u2;

	  reig[SHOCKRGHT].y = u1+a;
	  reig[SHOCKRGHT].z = u2;
       }
       else if( dir==1 ){
	  reig[SHOCKLEFT].z = u1-a;
	  reig[SHOCKLEFT].y = u2;

	  reig[SLOWWLEFT].z = 0.0;
	  reig[SLOWWLEFT].y = 1.0;

	  reig[CTENTROPY].z = u1;
	  reig[CTENTROPY].y = u2;

	  reig[SHOCKRGHT].z = u1+a;
	  reig[SHOCKRGHT].y = u2;
       }

       reig[SHOCKLEFT].x = 1.0;
       reig[SHOCKLEFT].w = ekin + a*a/g - a*u1;
       reig[SHOCKLEFT].x *= -hdai;
       reig[SHOCKLEFT].y *= -hdai;
       reig[SHOCKLEFT].z *= -hdai;
       reig[SHOCKLEFT].w *= -hdai;

       reig[SLOWWLEFT].x = 0.0;
       reig[SLOWWLEFT].w = u2;
       
       reig[CTENTROPY].x = 1.0;
       reig[CTENTROPY].w = ekin;
       
       reig[SHOCKRGHT].x = 1.0;
       reig[SHOCKRGHT].w = ekin + a*a/g + a*u1;
       reig[SHOCKRGHT].x *= hdai;
       reig[SHOCKRGHT].y *= hdai;
       reig[SHOCKRGHT].z *= hdai;
       reig[SHOCKRGHT].w *= hdai;

       }
    else
    {
       //// Primitive eigenvector

       if( dir==0 ){	
	  reig[SHOCKLEFT].y = -a;
	  reig[SHOCKLEFT].z = 0.0;

	  reig[SLOWWLEFT].y = 0.0;
	  reig[SLOWWLEFT].z = 1.0;

	  reig[SHOCKRGHT].y = a;
	  reig[SHOCKRGHT].z = 0.0;
       }
       else if( dir==1 ){
	  reig[SHOCKLEFT].z = -a;
	  reig[SHOCKLEFT].y = 0.0;

	  reig[SLOWWLEFT].z = 0.0;
	  reig[SLOWWLEFT].y = 1.0;

	  reig[SHOCKRGHT].z = a;
	  reig[SHOCKRGHT].y = 0.0;
       }

       reig[SHOCKLEFT].x = d;
       reig[SHOCKLEFT].w = d*a*a;

       reig[SLOWWLEFT].x = 0.0;
       reig[SLOWWLEFT].w = 0.0;

       reig[CTENTROPY].x = 1.0;
       reig[CTENTROPY].y = 0.0;
       reig[CTENTROPY].z = 0.0;
       reig[CTENTROPY].w = 0.0;

       reig[SHOCKRGHT].x = d;
       reig[SHOCKRGHT].y = a;
       reig[SHOCKRGHT].z = 0.0;
       reig[SHOCKRGHT].w = d*a*a;         
    }   
}

/*--------------- Left Eigenvalues --------------------*/
__host__ __device__ void left_eigenvectors(double4 V,bool conservative, double4 leig[NUMB_WAVE], int dir){ 
//Left Eigenvectors
    double  a, u1, u2,gi, d, g, ekin, a2inv, dinv, ahinv, Na;
    
    // velocity
    if (dir == 0){
	u1 = V.x;
	u2 = V.y;}
    if (dir == 1){
	u1 = V.y;
	u2 = V.x;}

    // sound speed, and others
    a = sqrt(sim_gamma*V.w/V.x);
    d = V.x;
    g = sim_gamma - 1.0;
    ekin = 0.5*(u1*u1+u2*u2);
    a2inv = 1.0/(a*a);
    Na = 0.5*a2inv;
    dinv = 1./d;
    ahinv = 0.5/a;
   
    if (conservative) {
       //// Conservative eigenvector
	
	if( dir==0 ){
       	  leig[SHOCKLEFT].y = -g*u1-a;
          leig[SHOCKLEFT].z = -g*u2;

	  leig[SLOWWLEFT].y = 0.0;
	  leig[SLOWWLEFT].z = 1.0;

	  leig[CTENTROPY].y = g*u1*a2inv;
	  leig[CTENTROPY].z = g*u2*a2inv;

	  leig[SHOCKRGHT].y = -g*u1+a;
	  leig[SHOCKRGHT].z = -g*u2;
	}
	else if( dir==1 ){
       	  leig[SHOCKLEFT].z = -g*u1-a;
          leig[SHOCKLEFT].y = -g*u2;

	  leig[SLOWWLEFT].z = 0.0;
	  leig[SLOWWLEFT].y = 1.0;

	  leig[CTENTROPY].z = g*u1*a2inv;
	  leig[CTENTROPY].y = g*u2*a2inv;

	  leig[SHOCKRGHT].z = -g*u1+a;
	  leig[SHOCKRGHT].y = -g*u2;
	}

       leig[SHOCKLEFT].x = -ekin - a*u1*gi;
       leig[SHOCKLEFT].w = -1.0;
       leig[SHOCKLEFT].x = Na*leig[SHOCKLEFT].x;
       leig[SHOCKLEFT].y = Na*leig[SHOCKLEFT].y;
       leig[SHOCKLEFT].z = Na*leig[SHOCKLEFT].z;
       leig[SHOCKLEFT].w = Na*leig[SHOCKLEFT].w;

       leig[SLOWWLEFT].x = -u2;
       leig[SLOWWLEFT].w = 0.0;

       leig[CTENTROPY].x = 1.0 -2.0*Na*g*ekin;
       leig[CTENTROPY].w = -g*a2inv;
       
       leig[SHOCKRGHT].x = g*ekin-u1*a;
       leig[SHOCKRGHT].w = g;
       leig[SHOCKRGHT].x = Na*leig[SHOCKRGHT].x;
       leig[SHOCKRGHT].y = Na*leig[SHOCKRGHT].y;
       leig[SHOCKRGHT].z = Na*leig[SHOCKRGHT].z;
       leig[SHOCKRGHT].w = Na*leig[SHOCKRGHT].w;

    }       
    else{
       if( dir==0 ){
	  leig[SHOCKLEFT].y = -ahinv;
	  leig[SHOCKLEFT].z = 0.0;

	  leig[SLOWWLEFT].y = 0.0;
	  leig[SLOWWLEFT].z = 1.0;

	  leig[SHOCKRGHT].y = ahinv;
	  leig[SHOCKRGHT].z = 0.0;
       }
       else if( dir==1 ){
	  leig[SHOCKLEFT].z = -ahinv;
	  leig[SHOCKLEFT].y = 0.0;

	  leig[SLOWWLEFT].z = 0.0;
	  leig[SLOWWLEFT].y = 1.0;
	
	  leig[SHOCKRGHT].z = ahinv;
	  leig[SHOCKRGHT].y = 0.0;

       }


       //// Primitive eigenvector
       leig[SHOCKLEFT].x = 0.0;
       leig[SHOCKLEFT].w = dinv*Na;

       leig[SLOWWLEFT].x = 0.0;
       leig[SLOWWLEFT].w = 0.0;

       leig[CTENTROPY].x = 1.0;
       leig[CTENTROPY].y = 0.0;
       leig[CTENTROPY].z = 0.0;
       leig[CTENTROPY].w = -a2inv;

       leig[SHOCKRGHT].x = 0.0;
       leig[SHOCKRGHT].w = dinv*Na;
    }
}

