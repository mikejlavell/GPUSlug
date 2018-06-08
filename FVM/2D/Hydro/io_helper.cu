#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <cmath>

#include "definition.h"
#include "io_helper.h"


/*--------------------- Function to export data to .dat file ----------------*/
 void io_writeOutput(int ioCounter, double *x, double *y, double4 *U)
 {  
    std::string sim;        
    std::ofstream myfile;
    if (sim_type == 1) sim = "Blast";
    else if(sim_type == 2) sim = "KelvinHelmoltz";
    else if(sim_type == 3) sim = "RMI1";
    else if(sim_type == 4) sim = "RMI2";
    else if(sim_type == 5) sim = "Double Mach Reflection";
    else if(sim_type == 6) sim = "Mach 3 Wind Tunnel";
    myfile.open("slug"+std::to_string(ioCounter)+sim+".dat");
    

    // 2D io
    for(int i = 0; i<GRID_XSZE; i++){
	for(int j = 0; j< GRID_YSZE; j++){
	    myfile << x[i]      <<'\t';  // x-position
	    myfile << y[j]      <<'\t';  // y-position
	    //myfile << U[j*M+i].x <<'\t';  // density
	    //myfile << U[j*M+i].y <<'\t';  // x-velocity
	    //myfile << U[j*M+i].z <<'\t';  // y-velocity
	    myfile << U[i*GRID_YSZE+j].w << std::endl;  // pressure
	}
	//myfile << std::endl;
	
    }
    //myfile << std::endl;
    myfile.close();

 }

