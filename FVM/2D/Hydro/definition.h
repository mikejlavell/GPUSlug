#ifndef definitions
#define definitions

//Simulation Parameters
 const double gr_xbeg = -1.0;
 const double gr_xend = 1.0;
 const double gr_ybeg = -1.0;
 const double gr_yend = 1.0;
 const int gr_ngc = 2;
 const int N = 128;
 const int M = 128;
 const int GRID_XSZE = N + 2*gr_ngc;
 const int GRID_YSZE = M + 2*gr_ngc;
 const double gr_dx = (gr_xend - gr_xbeg)/(GRID_XSZE-1);
 const double gr_dy = (gr_yend - gr_ybeg)/(GRID_YSZE-1);
 const int gr_ibeg = gr_ngc; //0 is in the gaurd cells
 const int gr_jbeg = gr_ngc; 
 const int gr_iend = N + gr_ibeg - 1;
 const int gr_jend = M + gr_jbeg - 1;
 const int gr_imax = GRID_XSZE - 1;	//Because C/C++
 const int gr_jmax = GRID_YSZE - 1;
 const int sim_type = 1; // 2 3 4      1 = Blast
#define sim_gamma 1.4
#define sim_cfl 0.8
#define sim_riemann "roe" // "hll"; "hllc";
 const double sim_tmax = 0.20;
 const double sim_cdf = 0.8;

//CUDA Parameters
 const double BLOCK_DIMX = 16;
 const double BLOCK_DIMY = 16;

// slope limiters
 const int MINMOD = 1;
 const int MC     = 2;
 const int VANLEER= 3;
 const int sim_limiter = MINMOD;

// primitive vars
 const int DENS_VAR =1;
 const int VELX_VAR =2;
 const int VELY_VAR =3;
 const int PRES_VAR =4;

// conservative vars
 const int MOMX_VAR =2;
 const int MOMY_VAR =3;
 const int ENER_VAR =4;

// waves
 const int SHOCKLEFT =1;
 const int SLOWWLEFT =2;
 const int CTENTROPY =3;
 const int SHOCKRGHT =4;
 const int NUMB_WAVE =4;

// BC
 const int OUTFLOW  =1;
 const int PERIODIC =2;
 const int REFLECT  =3;
 const int USER     =4;
 const int sim_bcType = OUTFLOW;

 const double pi = 4.0*atan(1.0);
#endif
