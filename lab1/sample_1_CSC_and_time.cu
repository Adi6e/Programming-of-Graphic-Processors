#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define sqr(x) ((x)*(x))

#define CSC(call) 					\
do { 							\
	cudaError_t status = call;			\
	if (status != cudaSuccess) {							\
		fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));		\
		exit(0);								\
	}										\
} while(0)

__global__ void kernel(float *arr, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	while (idx < n) {
		assert(idx < n);
		arr[idx] = __sinf(sqr(arr[idx]));
		idx += offset;
	}
}

int main() {
	long int i, n = 100000000;
	float *arr = (float *)malloc(sizeof(float) * n);
	for(i = 0; i < n; i++)
		arr[i] = i / (float)(n - 1);
	float *dev_arr;
	
	CSC(cudaMalloc(&dev_arr, sizeof(float) * n));
	CSC(cudaMemcpy(dev_arr, arr, sizeof(float) * n, cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));
	CSC(cudaEventRecord(start));
	
	kernel<<<256, 256>>>(dev_arr, n);
	CSC(cudaDeviceSynchronize());
	CSC(cudaGetLastError());

	CSC(cudaEventRecord(stop));
	CSC(cudaEventSynchronize(stop));
	float t;
	CSC(cudaEventElapsedTime(&t, start, stop));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(stop));

	printf("time = %f ms\n", t);

	CSC(cudaMemcpy(arr, dev_arr, sizeof(float) * n, cudaMemcpyDeviceToHost));
	for(i = n - 10; i < n; i++)
		printf("%f ", arr[i]);
	printf("\n");

	CSC(cudaFree(dev_arr));
	free(arr);
	return 0;
}
