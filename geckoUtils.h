
#ifndef __GECKO_UTILS_H__
#define __GECKO_UTILS_H__

#if defined(DEBUG) || defined(INFO)

#define GECKO_CUDA_CHECK(command)																	\
{																									\
	cudaError_t status = (command);                                                             	\
	if (status != cudaSuccess) {                                                                   	\
		fprintf(stderr, "\nCUDA Error: %s - Line %d - %s (code: %d): %s\n", __FILE__, __LINE__,     \
			cudaGetErrorName(status), status, cudaGetErrorString(status));    						\
	}																								\
}

#else

//#define CHAM_CUDA_CHECK(command)	command

#define GECKO_CUDA_CHECK(command)																	\
{																									\
	cudaError_t status = (command);                                                             	\
	if (status != cudaSuccess) {                                                                   	\
		exit(1);                                                                                    \
	}																								\
}

#endif

#endif //__GECKO_UTILS_H__

