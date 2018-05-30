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
	u = V.x;
    else if(dir == 2)
	u = V.y;

    // sound speed
    a = sqrt(sim_gamma*V.z/V.w);
    
    lambda[SHOCKLEFT] = u - a;
    lambda[CTENTROPY] = u;
    lambda[SHOCKRGHT] = u + a;
}



  
  void right_eigenvectors(double4 V,bool conservative, double3 reig[NUMB_WAVE], int dir){
  //Right Eigenvectors

    double  a, u, d, g, ekin, hdai, hda;

    // velocity
    if (dir == 1)
	u = V.x;
    else if(dir == 2)
	u = V.y;
    
    // sound speed, and others
    a = sqrt(sim_gamma*V.z/V.w);
    d = V.z;
    g = sim_gamma - 1.0;
    ekin = 0.5*u*u;
    hdai = 0.5*d/a;
    hda  = 0.5*d*a;
    
    if (conservative){
       //// Conservative eigenvector
       reig.x[SHOCKLEFT] = 1.0;
       reig.y[SHOCKLEFT] = u - a;
       reig.z[SHOCKLEFT] = ekin + a*a/g - a*u;
       reig.x[SHOCKLEFT] /= -hdai;
       reig.y[SHOCKLEFT] /= -hdai;
       reig.z[SHOCKLEFT] /= -hdai;

       reig.x[CTENTROPY] = 1.0;
       reig.y[CTENTROPY] = u;
       reig.z[CTENTROPY] = ekin;
       
       reig.x[SHOCKRGHT] = 1.0;
       reig.y[SHOCKRGHT] = u + a;
       reig.z[SHOCKRGHT] = ekin + a*a/g + a*u;
       reig.x[SHOCKRGHT] /= hdai;
       reig.y[SHOCKRGHT] /= hdai;
       reig.z[SHOCKRGHT] /= hdai;

       }
    else
    {
       //// Primitive eigenvector
       reig.x[SHOCKLEFT] = -hdai;
       reig.y[SHOCKLEFT] = 0.5;
       reig.z[SHOCKLEFT] = -hda;

       reig.x[CTENTROPY] = 1.0;
       reig.y[CTENTROPY] = 0.0;
       reig.z[CTENTROPY] = 0.0;

       reig.x[SHOCKRGHT] = hdai;
       reig.y[SHOCKRGHT] = 0.5;
       reig.z[SHOCKRGHT] = hda;         
    }   
}


void left_eigenvectors(double4 V,bool conservative, double3 leig[NUMB_WAVE], int dir){ 
//Left Eigenvectors
    double  a, u, d, g, gi, ekin, hdai, hda;
    
    // velocity
    if (dir == 0)
	u = V.x;
    if (dir == 1)
	u = V.y;

    // sound speed, and others
    a = sqrt(sim_gamma*V.z/V.x);
    d = V.z;
    g = sim_gamma - 1.0;
    ekin = 0.5*u*u;
    hdai = 0.5*d/a;
    hda  = 0.5*d*a;
    
    if (conservative) {
       //// Conservative eigenvector
       leig.x[SHOCKLEFT] = -ekin - a*u*gi;
       leig.y[SHOCKLEFT] = u+a*gi;
       leig.z[SHOCKLEFT] = -1.0;
       leig.x[SHOCKLEFT] = g*leig.x[SHOCKLEFT]/(d*a);
       leig.y[SHOCKLEFT] = g*leig.y[SHOCKLEFT]/(d*a);
       leig.z[SHOCKLEFT] = g*leig.z[SHOCKLEFT]/(d*a);

       leig.x[CTENTROPY] = d*(-ekin + gi*a*a)/a;
       leig.y[CTENTROPY] = d*u/a;
       leig.z[CTENTROPY] = -d/a;
       leig.x[CTENTROPY] = g*leig.x[CTENTROPY]/(d*a);
       leig.y[CTENTROPY] = g*leig.y[CTENTROPY]/(d*a);
       leig.z[CTENTROPY] = g*leig.z[CTENTROPY]/(d*a);
       
       leig.x[SHOCKRGHT] = ekin - a*u*gi;
       leig.y[SHOCKRGHT] = -u+a*gi;
       leig.z[SHOCKRGHT] = 1.0;
       leig.x[SHOCKRGHT] = g*leig.x[SHOCKRGHT]/(d*a)
       leig.y[SHOCKRGHT] = g*leig.y[SHOCKRGHT]/(d*a)
       leig.z[SHOCKRGHT] = g*leig.z[SHOCKRGHT]/(d*a)

    }       
    else
    {
       //// Primitive eigenvector
       leig.x[SHOCKLEFT] = 0.0;
       leig.y[SHOCKLEFT] = 1.0;
       leig.z[SHOCKLEFT] = -1.0/(d*a);

       leig.x[CTENTROPY] = 1.0;
       leig.y[CTENTROPY] = 0.0;
       leig.z[CTENTROPY] = -1.0/(a*a);

       leig.x[SHOCKRGHT] = 0.0;
       leig.y[SHOCKRGHT] = 1.0;
       leig.z[SHOCKRGHT] = 1.0/(d*a);
    }
}

