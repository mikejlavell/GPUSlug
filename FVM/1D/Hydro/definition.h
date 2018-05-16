//Simulation Parameters
 double gr_xbeg = 0.0;
 double gr_xend = 1.0;
 int gr_ngc = 2;
 int N = 1024;
 int GRID_SIZE = N + 2*gr_ngc;
 double gr_dx = (gr_xend - gr_xbeg)/GRID_SIZE;
 int gr_ibeg = 2; //0 is in the gaurd cells
 int gr_iend = N + gr_ibeg - 1;
 int gr_imax = GRID_SIZE -1;	//Because C/C++
 int sim_type = 1; // 2 3 4      1 = Sod, 2 = Rarefaction, 3 = Blast2, 4 = ShuOsher
 double sim_gamma = 1.4;
 std::string sim_riemann = "roe"; // "hll"; "hllc";
 double sim_tmax = 0.20;

// slope limiters
 int MINMOD = 1;
 int MC     = 2;
 int VANLEER= 3;
 int slopelimiter = MINMOD;

// primitive vars
 int DENS_VAR =1;
 int VELX_VAR =2;
 int PRES_VAR =3;

// conservative vars
 int MOMX_VAR =2;
 int ENER_VAR =3;

// waves
 int SHOCKLEFT =1;
 int CTENTROPY =2;
 int SHOCKRGHT =3;
 int NUMB_WAVE =3;

// BC
 int OUTFLOW  =1;
 int PERIODIC =2;
 int REFLECT  =3;
 int USER     =4;
 int sim_bcType = OUTFLOW;

 double pi = 4.0*atan(1.0);
