#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>

#include "definition.h"

/*--------------------- Function to export data to .dat file ----------------*/
 void io_writeOutput(int ioCounter, double *x, double3 *U)
 {  
    std::string sim;        
    std::ofstream myfile;
    if (sim_type == 1) sim = "Sod";
    else if(sim_type == 2) sim = "Rare";
    else if(sim_type == 3) sim = "Blast2";
    else if(sim_type == 4) sim = "ShuOsher";
    myfile.open("slug"+std::to_string(ioCounter)+sim+".dat");
    
    for(int j = 0; j<GRID_SIZE; j++){
        myfile << x[j] << '\t';
    }
    myfile << std::endl;

    for(int j = 0; j<GRID_SIZE; j++){
        myfile << U.x[j] << '\t';
    }
    myfile << std::endl;
    for(int j = 0; j<GRID_SIZE; j++){
        myfile << U.y[j] << '\t';
    }
    myfile << std::endl;
    for(int j = 0; j<GRID_SIZE; j++){
        myfile << U.z[j] << '\t';
    }
    myfile << std::endl;
    myfile.close();
 }
