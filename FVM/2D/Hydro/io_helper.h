#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>

#include "definition.h"

/*--------------------- Function to export data to .dat file ----------------*/
 void io_writeOutput(int ioCounter, double *x, double *y, double3 *U)
 {  
    std::string sim;        
    std::ofstream myfile;
    if (sim_type == 1) sim = "Blast";
    else if(sim_type == 2) sim = "KelvinHelmoltz";
    else if(sim_type == 3) sim = "RMI1";
    else if(sim_type == 4) sim = "RMI2";
    else if(sim_type == 5) sim = "Double Mach Reflection
    else if(sim_type == 6) sim = "Mach 3 Wind Tunnel
    myfile.open("slug"+std::to_string(ioCounter)+sim+".dat");
    

    // 2D io
    double xx = x[gr_ibeg:gr_iend];
    double yy = y[gr_jbeg:gr_jend];

    double dens = U.x[gr_ibeg:gr_iend][gr_jbeg:gr_jend];
    double velx = U.y[gr_ibeg:gr_iend][gr_jbeg:gr_jend];
    double vely = U.z[gr_ibeg:gr_iend][gr_jbeg:gr_jend];
    double pres = U.w[gr_ibeg:gr_iend][gr_jbeg:gr_jend];
   
    for(int i = gr_ibeg; i<gr_iend; i++){
	for(int j = gr_jbeg; j< gr_jend; j++){
	    myfile << xx[i] <<'\t'<< yy[j] <<'\t'<< U.x[i][j] >>'\t'<< U.y[i][j] \\
		   <<'\t'<< U.z[i][j] <<'\t'<< U.w[i][j] << '\t';

	}
    }
    myfile << std::endl;
    myfile.close();

 }
