############################# Compiler and Flags #################
C = g++ 
NVCC = nvcc 
NVCCFLAGS = -arch=sm_60
CFLAGS= -std=c++11
############################## Object Declaration ################

EXE_FILE: slug_1d.exe
OBJS = driver_euler1D.o \
soln_update.o \ 
roe.o 

############################# Compilation and Linking #############
	
$(EXE_FILE): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) $(OBJS) -o $(EXE_FILE)  

.SUFFIXES:
.SUFFICES: .cu .o
.cu.o: 
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) -c $<

driver_euler1D.o: soln_update.o
soln_update.o: roe.o 

############################# Clean ###############################
clean: 
	rm -f slug1d.exe *.o

