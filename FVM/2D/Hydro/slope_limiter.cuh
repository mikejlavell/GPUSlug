#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>
#include <algorithm>

/*Slope Limiters to run on the device*/
__device__ double vanleer(double a, double b){//VanLeer Slope Limiter
    double delta;
    if (a*b > 0) delta = 2.0*a*b/(a+b);
    else delta = 0.0;
    return delta;
}

__device__ double minmod(double a,double b){//Minmod Slope Limiter
   double delta = 0.5 * (sign(1.0,a) + sign(1.0,b))*min(abs(a),abs(b));
   return delta;
}

  
__device__ double mc(double a,double b){//MC slope limiter
   double delta = (sign(1.0,a)+sign(1.0,b))*min(abs(a),0.25*abs(a+b),abs(b));
   return delta;
}

