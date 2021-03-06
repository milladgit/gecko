
#include <stdio.h>


int main() {

	#pragma gecko loctype name("host") kind("x64", "Skylake") num_cores(4) mem("4MB") 
	#pragma gecko loctype name("tesla") kind("NVIDIA", "cc50") mem("4GB")
	#pragma gecko loctype name("NODE_MEMORY") kind("Unified_Memory") size("16GB")


	#pragma gecko location name("LocA") type("NODE_MEMORY")
	#pragma gecko location name("LocN","LocG") type("virtual")
//	#pragma gecko location name("LocN1", "LocN2") type("host")
	#pragma gecko location name("LocN1") type("host")
	#pragma gecko location name("LocG1") type("tesla")

	char op = '+';
	#pragma gecko hierarchy children(op:"LocN","LocG") parent("LocA")
	//#pragma gecko hierarchy children(op:"LocN1","LocN2") parent("LocN")
	#pragma gecko hierarchy children(op:"LocN1") parent("LocN")
	#pragma gecko hierarchy children(+:"LocG1") parent("LocG")


	#pragma gecko draw root("LocA")


	int N = 2000;
	double *X, *Y, *Z, *W;
	#pragma gecko memory allocate(X[0:N]) type(double) location("LocN") 
	#pragma gecko memory allocate(Y[0:N]) type(double) location("LocG") 
	#pragma gecko memory allocate(Z[0:N]) type(double) location("LocA") 
	#pragma gecko memory allocate(W[0:N]) type(double) location("LocA") 

	gecko_double Q;
	#pragma gecko memory allocate(Q[0:N]) type(gecko_double) location("LocA") 

	// #pragma gecko memory allocate(T[0:N]) type(double) distance(near)


	for (int i = 0; i<N; i++) {
		Z[i] = 0.0;
		Q[i] = i;
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


	double coeff = 3.4;
	for(int q=0;q<2;q++) {
		a = 0;
		b = N;
		#pragma gecko region at("LocA") exec_pol("static") variable_list(Z) variable_list_internal(Q)
		// #pragma gecko region at("LocA") exec_pol("any") variable_list(Z)
		#pragma acc parallel loop present(Z)
		for (int i = a; i<b; i++) {
			Z[i] = Q[i] * coeff;
		}
		#pragma gecko region end

	}
	#pragma gecko region pause at("LocA") 



	// #pragma acc wait



	printf("Checking...\n");
	int c = 0;
	for(int i=0;i<N;i++) {
		if(Z[i] != (Q[i]*coeff)) {
			printf("Error in index: %d - V: %.2f\n", i, Z[i]);
			c++;
			if(c>=10) break;
		}
	}
	printf("Checking...Done\n");


	printf("Hello World!\n");


    for(int i=980;i<1020;i++)
        printf("Q[%d]: %2f\n", i, Q[i]);


	#pragma gecko memory free(X)
	#pragma gecko memory free(Y,Z,)
	#pragma gecko memory freeobj(Q)

	return 0;
}
