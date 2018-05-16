#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>
#include <algorithm>

/*Slope Limiters to run on the device*/
__device__ float sign(float a, float b)
{	
	float temp; 
	if(b < 0.0f) temp = -fabs(a);
	else if(b > 0.0f) temp = fabs(a);
	else if(b == 0.0f) temp = a;
	return temp;
		
}

__device__ float vanLeer(float a, float b){//VanLeer Slope Limiter
    float delta;
    if (a*b > 0.0f) delta = 2.0f*a*b/(a+b);
    else delta = 0.0f;
    return delta;
}

__device__ float minmod(float a,float b){//Minmod Slope Limiter
   float delta = 0.5f * (sign(1.0f,a) + sign(1.0f,b))*fmin(abs(a),abs(b));
   return delta;
}

  
__device__ float mc(float a,float b){//MC slope limiter
   float temp = fmin(fabs(a), 0.25f*fabs(a + b)); 
   return (sign(1.0f,a)+sign(1.0f,b))*fmin(temp,fabs(b));;
}

