#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define CSC(call) 					\
do { 							\
	cudaError_t status = call;			\
	if (status != cudaSuccess) {							\
		fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));		\
		exit(0);								\
	}										\
} while(0)

__global__ void kernel(float *first_vec, float *second_vec, float *res_vec, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	while (idx < n) 
    {
        if (first_vec[idx] > second_vec[idx])
            res_vec[idx] = first_vec[idx];
        else
            res_vec[idx] = second_vec[idx];
        idx += offset;
	}
}

int main() 
{
	int i, n;
    scanf("%d", &n);

	float *first_arr = (float *)malloc(sizeof(float) * n);
	float *second_arr = (float *)malloc(sizeof(float) * n);
    float *res_arr = (float *)malloc(sizeof(float) * n);

	for(i = 0; i < n; i++)
		scanf("%f", &first_arr[i]);
	for(i = 0; i < n; i++)
		scanf("%f", &second_arr[i]);
	for(i = 0; i < n; i++)
		res_arr[i] = i;

	float *dev_arr1, *dev_arr2, *dev_arr3;
	CSC(cudaMalloc(&dev_arr1, sizeof(float) * n));
	CSC(cudaMemcpy(dev_arr1, first_arr, sizeof(float) * n, cudaMemcpyHostToDevice));
	CSC(cudaMalloc(&dev_arr2, sizeof(float) * n));
	CSC(cudaMemcpy(dev_arr2, second_arr, sizeof(float) * n, cudaMemcpyHostToDevice));
	CSC(cudaMalloc(&dev_arr3, sizeof(float) * n));
	CSC(cudaMemcpy(dev_arr3, res_arr, sizeof(float) * n, cudaMemcpyHostToDevice));

	kernel<<<32, 32>>>(dev_arr1, dev_arr2, dev_arr3, n);
	CSC(cudaDeviceSynchronize());
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(res_arr, dev_arr3, sizeof(float) * n, cudaMemcpyDeviceToHost));

	for(i = 0; i < n; i++)
		printf("%.10e ", res_arr[i]);
	printf("\n");

	CSC(cudaFree(dev_arr1));
    CSC(cudaFree(dev_arr2));
    CSC(cudaFree(dev_arr3));
	free(first_arr);
    free(second_arr);
    free(res_arr);
	return 0;
}
