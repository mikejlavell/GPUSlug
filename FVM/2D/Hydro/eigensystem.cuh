#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>

extern double sim_gamma;
extern int SHOCKLEFT;
extern int CTENTROPY;
extern int SHOCKRGHT;
extern int NUMB_WAVE;

 void eigenvalues(double4 V,double lambda[NUMB_WAVE], int dir){
    double  a, u;

    // velocity
    if (dir == 1)
	u = V.y;
    else if(dir == 2)
	u = V.z;

    // sound speed
    a = sqrtf(sim_gamma*V.w/V.x);
    
    lambda[SHOCKLEFT] = u - a;
    lambda[CTENTROPY] = u;
    lambda[SHOCKRGHT] = u + a;
}



  
  void right_eigenvectors(double4 V, bool conservative, double4 reig[NUMB_WAVE], int dir){
  //Right Eigenvectors

    double  a, u, d, g, ekin, hdai, hda;

    // velocity
    if (dir == 1)
	u1 = V.y;
	u2 = V.z;
    else if(dir == 2)
	u1 = V.z;
	u2 = V.y;
    
    // sound speed, and others
    a = sqrt(sim_gamma*V.w/V.x);
    d = V.x;
    g = sim_gamma - 1.0;
    ekin = 0.5*(u1*u1 + u2*u2);
    hdai = 0.5*d/a;
    hda  = 0.5*d*a;
    
    if (conservative){
       //// Conservative eigenvector
       reig[SHOCKLEFT].x = 1.0;
       reig[SHOCKLEFT].y = u1 - a;
       reig[SHOCKLEFT].z = u2;
       reig[SHOCKLEFT].w = ekin + a*a/g - a*u;
       reig[SHOCKLEFT].x *= -hdai;
       reig[SHOCKLEFT].y *= -hdai;
       reig[SHOCKLEFT].z *= -hdai;
       reig[SHOCKLEFT].w *= -hdai;

       reig[CTENTROPY].x = 1.0;
       reig[CTENTROPY].y = u1;
       reig[CTENTROPY].z = u2;
       reig[CTENTROPY].w = ekin;
       
       reig[SHOCKRGHT].x = 1.0;
       reig[SHOCKRGHT].y = u1 + a;
       reig[SHOCKRGHT].z = u2;
       reig[SHOCKRGHT].w = ekin + a*a/g + a*u;
       reig[SHOCKRGHT].x *= hdai;
       reig[SHOCKRGHT].y *= hdai;
       reig[SHOCKRGHT].z *= hdai;
       reig[SHOCKRGHT].w *= hdai;

       }
    else
    {
       //// Primitive eigenvector
       reig[SHOCKLEFT].x = -hdai;
       reig[SHOCKLEFT].y = 0.5;
       reig[SHOCKLEFT].z = 0.0;
       reig[SHOCKLEFT].w = -hda;

       reig[CTENTROPY].x = 1.0;
       reig[CTENTROPY].y = 0.0;
       reig[CTENTROPY].z = 0.0;
       reig[CTENTROPY].w = 0.0;

       reig[SHOCKRGHT].x = hdai;
       reig[SHOCKRGHT].y = 0.5;
       reig[SHOCKRGHT].z = 0.0;
       reig[SHOCKRGHT].w = hda;         
    }   
}


void left_eigenvectors(double4 V,bool conservative, double3 leig[NUMB_WAVE], int dir){ 
//Left Eigenvectors
    double  a, u, d, g, gi, ekin, hdai, hda;
    
    // velocity
    if (dir == 0)
	u1 = V.x;
	u2 = V.y;
    if (dir == 1)
	u1 = V.y;
	u2 = V.x

    // sound speed, and others
    a = sqrt(sim_gamma*V.w/V.x);
    d = V.x;
    g = sim_gamma - 1.0;
    ekin = 0.5*(u1*u1+u2*u2);
    hdai = 0.5*d/a;
    hda  = 0.5*d*a;
    
    if (conservative) {
       //// Conservative eigenvector
       leig[SHOCKLEFT].x = -ekin - a*u1*gi;
       leig[SHOCKLEFT].y = u1+a*gi;
       leig[SHOCKLEFT].z = u2;
       leig[SHOCKLEFT].w = -1.0;
       leig[SHOCKLEFT].x = g*leig[SHOCKLEFT].x/(d*a);
       leig[SHOCKLEFT].y = g*leig[SHOCKLEFT].y/(d*a);
       leig[SHOCKLEFT].z = g*leig[SHOCKLEFT].z/(d*a);
       leig[SHOCKLEFT].w = g*leig[SHOCKLEFT].w/(d*a);

       leig[CTENTROPY].x = d*(-ekin + gi*a*a)/a;
       leig[CTENTROPY].y = d*u1/a;
       leig[CTENTROPY].z = d*u2/a;
       leig[CTENTROPY].w = -d/a;
       leig[CTENTROPY].x = g*leig[CTENTROPY].x/(d*a);
       leig[CTENTROPY].y = g*leig[CTENTROPY].y/(d*a);
       leig[CTENTROPY].z = g*leig[CTENTROPY].z/(d*a);
       leig[CTENTROPY].w = g*leig[CTENTROPY].w/(d*a);
       
       leig[SHOCKRGHT].x = ekin - a*u*gi;
       leig[SHOCKRGHT].y = -u1+a*gi;
       leig[SHOCKRGHT].z = -u2;
       leig[SHOCKRGHT].w = 1.0;
       leig[SHOCKRGHT].x = g*leig[SHOCKRGHT].x/(d*a);
       leig[SHOCKRGHT].y = g*leig[SHOCKRGHT].y/(d*a);
       leig[SHOCKRGHT].z = g*leig[SHOCKRGHT].z/(d/a);
       leig[SHOCKRGHT].w = g*leig[SHOCKRGHT].w/(d*a);

    }       
    else
    {
       //// Primitive eigenvector
       leig[SHOCKLEFT].x = 0.0;
       leig[SHOCKLEFT].y = 1.0;
       leig[SHOCKLEFT].z = 0.0;
       leig[SHOCKLEFT].w = -1.0/(d*a);

       leig[CTENTROPY].x = 1.0;
       leig[CTENTROPY].y = 0.0;
       leig[CTENTROPY].z = 0.0;
       leig[CTENTROPY].w = -1.0/(a*a);

       leig[SHOCKRGHT].x = 0.0;
       leig[SHOCKRGHT].y = 1.0;
       leig[SHOCKRGHT].z = 0.0;
       leig[SHOCKRGHT].w = 1.0/(d*a);
    }
}

