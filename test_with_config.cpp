
#include <stdio.h>


int main(int argc, char **argv) {

	#pragma gecko config env
 
// 	#pragma gecko loctype name("host") kind("x64", "Skylake") num_cores(4) mem("4MB") 
// 	#pragma gecko loctype name("tesla") kind("CC3.0", "Volta") mem("4GB")
// 	#pragma gecko loctype name("NODE_MEMORY") kind("Unified_Memory") size("16GB")


// 	#pragma gecko location name("LocA") type("NODE_MEMORY")
// 	#pragma gecko location name("LocN","LocG") type("virtual")
// //	#pragma gecko location name("LocN1", "LocN2") type("host")
// 	#pragma gecko location name("LocN1") type("host")
// 	#pragma gecko location name("LocG1") type("tesla")

// 	char op = '+';
// 	#pragma gecko hierarchy children(op:"LocN","LocG") parent("LocA")
// 	//#pragma gecko hierarchy children(op:"LocN1","LocN2") parent("LocN")
// 	#pragma gecko hierarchy children(op:"LocN1") parent("LocN")
// 	#pragma gecko hierarchy children(+:"LocG1") parent("LocG")


	#pragma gecko draw root("LocA")


	int N = atoi(argv[1]);
	double *X, *Y, *Z;
	#pragma gecko memory allocate(X[0:N]) type(double) location("LocA") 
	#pragma gecko memory allocate(Y[0:N]) type(double) location("LocG") 
	#pragma gecko memory allocate(Z[0:N]) type(double) location("LocA") 


	for (int i = 0; i<N; i++) {
		Z[i] = 0.0;
	}


	int a, b;

	// a = 1821;
	// b = 23;

	// #pragma gecko region at("LocG") exec_pol("flatten") variable_list(Y)
	// // #pragma gecko region at(loc_name) exec_pol(exec_pol) variable_list(Y)
	// #pragma acc parallel loop 
	// for (int i = a; i>=b; i--) {
	// 	Y[i] *= 2.0;
	// }
	// #pragma gecko region end


	// a = 0;
	// b = N;
	// #pragma gecko region at("LocN") exec_pol("flatten") variable_list(X)
	// // #pragma gecko region at(loc_name) exec_pol(exec_pol) variable_list(Y)
	// #pragma acc parallel loop 
	// for (int i = a; i<b; i++) {
	// 	X[i] *= 2.0;
	// }
	// #pragma gecko region end



	// a = 0;
	// b = N;
	// #pragma gecko region at("LocG") exec_pol("static") variable_list(Z)
	// // #pragma gecko region at(loc_name) exec_pol(exec_pol) variable_list(Y)
	// #pragma acc parallel loop 
	// for (int i = a; i<b; i++) {
	// 	Z[i] = 2.0;
	// }
	// #pragma gecko region end


	a = 0;
	b = N;
	double coeff = 3.4;
	#pragma gecko region at("LocA") exec_pol("static") variable_list(Z)
	//#pragma gecko region at("LocA") exec_pol("any") variable_list(Z)
	#pragma acc parallel loop 
	for (int i = a; i<b; i++) {
		Z[i] = coeff * i;
	}
	#pragma gecko region end

	#pragma gecko region pause at("LocA") 

	// #pragma acc wait


#if 0
	printf("Checking...\n");
	for(int i=0;i<N;i++) {
		if(Z[i] != (coeff*i)) {
			printf("Error in index: %d\n", i);
		}
	}
	printf("Checking...Done\n");

#endif
	printf("Hello World!\n");

	#pragma gecko memory free(X)
	#pragma gecko memory free(Y)
	#pragma gecko memory free(Z)

	return 0;
}
