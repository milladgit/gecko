
#include <stdio.h>

int main(int argc, char **argv) {

	#pragma gecko loctype name("host") kind("x64", "Skylake") num_cores(4) size("4MB") 
	#pragma gecko loctype name("tesla") kind("NVIDIA", "cc50") size("4GB")
	#pragma gecko loctype name("NODE_MEMORY") kind("Unified_Memory") size("16GB")
	#pragma gecko loctype name("HDD") kind("Permanent_Storage") 


	#pragma gecko location name("LocA") type("NODE_MEMORY")
	#pragma gecko location name("LocN","LocG") type("virtual")
//	#pragma gecko location name("LocN1", "LocN2") type("host")
	#pragma gecko location name("LocN1") type("host")
	#pragma gecko location name("LocG1","LocG2") type("tesla")
	#pragma gecko location name("LocHDD") type("HDD")


	char op = '+';
	#pragma gecko hierarchy children(op:"LocA") parent("LocHDD")
	#pragma gecko hierarchy children(op:"LocN","LocG") parent("LocA")
	//#pragma gecko hierarchy children(op:"LocN1","LocN2") parent("LocN")
	#pragma gecko hierarchy children(op:"LocN1") parent("LocN")
	#pragma gecko hierarchy children(+:"LocG1","LocG2") parent("LocG")


	#pragma gecko draw root("LocA")


	int N = 2000;
	double *X, *Y, *Z, *W;
	#pragma gecko memory allocate(X[0:N]) type(double) location("LocN") 
	#pragma gecko memory allocate(Y[0:N]) type(double) location("LocG") 
	#pragma gecko memory allocate(Z[0:N]) type(double) location("LocA") 
	#pragma gecko memory allocate(W[0:N]) type(double) location("LocA") 

	double *Q;
	#pragma gecko memory allocate(Q[0:N]) type(double) location("LocG1") 

	// #pragma gecko memory copy from(X[0:N]) to(Y[0:N])
	// #pragma gecko memory move(Q) to("LocA")

	// vector<double> v(100);
	// double *v_addr = (double*) v.data();
	// #pragma gecko memory register(v_addr[0:100]) type(double) loc("LocN")
	// #pragma gecko memory unregister(v_addr) 

	double *T1, *T1_realloc, *T1_auto, *T2, *T2_far2, *T2_far_variable;
	#pragma gecko memory allocate(T1[0:N]) type(double) distance(near)
	#pragma gecko memory allocate(T1_realloc[0:N]) type(double) distance(near) realloc
	// #pragma gecko memory allocate(T1_auto[0:N]) type(double) distance(near) auto
	#pragma gecko memory allocate(T2[0:N]) type(double) distance(far) file("T2.obj")
	#pragma gecko memory allocate(T2_far2[0:N]) type(double) distance(far:2) file("T2_far.obj")
	int far_distance = 10;
	#pragma gecko memory allocate(T2_far_variable[0:N]) type(double) distance(far:far_distance) file("T2_far_variable.obj")


	double *Perm;
	#pragma gecko memory allocate(Perm[0:N]) type(double) location("LocHDD") file("perm.obj")



	// #pragma gecko region exec_pol("static") variable_list(Z,Q)
	// #pragma acc parallel loop present(Z)
	// for (int i = 0; i<N; i++) {
	// 	Z[i] = 0.0;
	// 	Q[i] = i;
	// }
	// #pragma gecko region end

	// #pragma gecko memory allocate(A_far[0:b-a]) type(double) distance(far:1)
	// #pragma gecko memory allocate(A_near[0:b-a]) type(double) distance(near)


	int a, b;

	double coeff = 3.4;
	for(int q=0;q<1;q++) {
		a = 0;
		b = N;
		long total = 0;
		#pragma gecko region exec_pol("static") variable_list(Perm,Z,Q,T1,T2_far_variable) reduction(+:total)
		// #pragma gecko region exec_pol("static") variable_list(Perm,Z,Q) reduction(+:total)
		for (int i = a; i<b; i++) {
			Z[i] = Q[i] * coeff;
			T1[i] *= 2;
			T2_far_variable[i] *= 2;
			total += (i+1);
			Perm[i] = i;
		}
		#pragma gecko region end

		printf("Total is %ld\n", total);
	}
	#pragma gecko region pause at("LocA") 


	printf("Checking...\n");
	int c = 0;
	// for(int i=0;i<N;i++) {
	// 	if(Z[i] != (Q[i]*coeff)) {
	// 		printf("Error in index: %d - V: %.2f\n", i, Z[i]);
	// 		c++;
	// 		if(c>=10) break;
	// 	}
	// }
	printf("Checking...Done\n");


	printf("Hello World!\n");


    // for(int i=980;i<1020;i++)
    //     printf("Q[%d]: %2f\n", i, Q[i]);


	#pragma gecko memory free(X,Y,Z,Q,W,Perm)

	return 0;
}
