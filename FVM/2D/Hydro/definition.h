//Simulation Parameters
 double gr_xbeg = 0.0;
 double gr_xend = 1.0;
 double gr_ybeg = 0.0;
 double gr_yend = 1.0;
 int gr_ngc = 2;
 int N = 32;
 int M = 32;
 int GRID_XSZE = N + 2*gr_ngc;
 int GRID_YSZE = M + 2*gr_ngc;
 double gr_dx = (gr_xend - gr_xbeg)/GRID_XSZE;
 double gr_dy = (gr_yend - gr_xbeg)/GRID_YSZE;
 int gr_ibeg = gr_ngc; //0 is in the gaurd cells
 int gr_jbeg = gr_ngc; 
 int gr_iend = N + gr_ibeg - 1;
 int gr_jend = M + gr_ibeg - 1;
 int gr_imax = GRID_XSZE - 1;	//Because C/C++
 int gr_jmax = GRID_YSZE - 1;
 int sim_type = 1; // 2 3 4      1 = Sod, 2 = Rarefaction, 3 = Blast2, 4 = ShuOsher
 double sim_gamma = 1.4;
 std::string sim_riemann = "roe"; // "hll"; "hllc";
 double sim_tmax = 0.20;

//CUDA Parameters
 double BLOCK_DIMX
 double BLOCK_DIMY

// slope limiters
 int MINMOD = 1;
 int MC     = 2;
 int VANLEER= 3;
 int slopelimiter = MINMOD;

// primitive vars
 int DENS_VAR =1;
 int VELX_VAR =2;
 int VELY_VAR =3;
 int PRES_VAR =4;

// conservative vars
 int MOMX_VAR =2;
 int MOMY_VAR =3
 int ENER_VAR =4;

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
