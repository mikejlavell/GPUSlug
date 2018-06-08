############################# Compiler and Flags #################
C = g++ 
NVCC = nvcc 
NVCCFLAGS = -arch=sm_30 -g
CFLAGS= -std=c++11

############################## Object Declaration ################
EXE_FILE = slug2d.exe
#DRIVER: driver_euler2D.cu
OBJS =  soln_update.o\
	soln_ReconEvolve.o\
	roe.o\
	io_helper.o\
	cfl.o\
	eos.o\
	eigensystem.o\
	primconsflux.o\
	slope_limiter.o\
	Slug_helper.o\
	driver_euler2D.o

############################# Compilation and Linking #############
	
$(EXE_FILE) : $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) $(OBJS) -o $(EXE_FILE)

.SUFFIXES:
.SUFFIXES: .cu .o

.cu.o: 
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) -c -I. -dc $< -o $@

.cpp.o:
	$(NVCC) $(NVCCFLAGS)  $(CFLAGS) -c $<

############################# Module dependencies ################
driver_euler2D.o : soln_update.o soln_ReconEvolve.o
soln_ReconEvolve.o : roe.o


############################# Clean ###############################
clean: 
	rm -f slug2d.exe *.o

