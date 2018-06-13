
#include <stdio.h>


int main() {
 
	#pragma gecko loctype name("host") kind("x64", "Skylake") num_cores(4) mem("4MB") 
	#pragma gecko loctype name("tesla") kind("CC3.0", "Volta") mem("4GB")
	#pragma gecko loctype name("NODE_MEMORY") kind("Unified_Memory") size("16GB")


	#pragma gecko location name("LocA") type("NODE_MEMORY")
	#pragma gecko location name("LocB","LocC") type("virtual")
	#pragma gecko location name("LocQ") type("host")
	#pragma gecko location name("LocW", "LocWWW", "LocX",  "LocY") type("virtual")
	#pragma gecko location name("LocD","LocE") type("host")
	// #pragma gecko location name("LocF") type("tesla")
	// #pragma gecko location name("LocG") all type("tesla")
	// #pragma gecko location name(loc_name) all type("tesla")
	// #pragma gecko location name("LocH[0:2]") type("multicore")
	// #pragma gecko location name("LocI[0:n]") type("multicore")

	char op = '+';
	#pragma gecko hierarchy children(op:"LocB","LocC") parent("LocA")
	#pragma gecko hierarchy children(op:"LocD","LocE") parent("LocB")
	// #pragma gecko hierarchy children(op:"LocQ") all parent("LocB")
	// #pragma gecko hierarchy children(op:"LocW") parent("LocB")
	// #pragma gecko hierarchy children(op:"LocX") parent("LocB")
	#pragma gecko hierarchy children(+:"LocWWW") parent("LocC")
	// #pragma gecko hierarchy children(-:"LocWWW") parent("LocC")
	#pragma gecko hierarchy children(+:"LocY") parent("LocC")
	#pragma gecko hierarchy children(+:"LocQ") parent("LocB")
	// #pragma gecko hierarchy children(-:"LocD","LocE") parent("LocB")
	// #pragma gecko hierarchy children(cmd:"LocF") parent("LocC")
	// #pragma gecko hierarchy children(cmd:"LocG") parent("LocC")
	// #pragma gecko hierarchy children(cmd:loc_name) all parent("LocC")
	// #pragma gecko hierarchy children(cmd:"LocI[2:4]") parent("LocC")


	#pragma gecko draw root("LocA")


	int N = 2000;
	double *X, *Y, *Y2, *Z, *YY, *ZZ;
	// #pragma gecko memory allocate(X[0:N]) type(double) location("LocA")
	// #pragma gecko memory allocate(X2) type(double) location("LocA")
	#pragma gecko memory allocate(Y[0:N]) type(double) location("LocB") 
	#pragma gecko memory allocate(Y2[0:N]) type(double) location("LocD") 
	// #pragma gecko memory allocate(Z[0:N]) type(double) location("LocC") 
	// #pragma gecko memory allocate(YY[0:N]) type(double) location("LocF") 
	// #pragma gecko memory allocate(ZZ[0:N]) type(double) location("LocD") 

	// // #pragma gecko put(A.X[0:N/2],F.YY[0:N/2]) 


	int a = 5000;
	int b = N;

	#pragma gecko region at("LocA") exec_pol("static") variable_list(Y)
	// #pragma gecko region at(loc_name) exec_pol(exec_pol) variable_list(Y)
	#pragma acc parallel loop 
	for (int i = a; i>=b; i--) {
		Y[i] *= 2.0;
	}
	#pragma gecko region end




	printf("Hello World!\n");

	return 0;
}
