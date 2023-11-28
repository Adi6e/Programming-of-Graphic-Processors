#include <stdio.h>
#include <math.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>


#define CSC(call) 					\
do { 							\
	cudaError_t status = call;			\
	if (status != cudaSuccess) {							\
		fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));		\
		exit(0);								\
	}										\
} while(0)

struct comparator {												
	__host__ __device__ bool operator()(double a, double b){
		return fabs(a) < fabs(b);
	}
};

__global__ void kernel_swap_rows(double *kernel_data, int n, int first, int second)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;

    while(idx < n)
    {
        double temp = kernel_data[first + idx * n];
        kernel_data[first + idx * n] = kernel_data[second + idx * n];
        kernel_data[second + idx * n] = temp;
        idx += offsetx;
    }
}

__global__ void kernel_gauss(double *kernel_data, int n, int ind)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x ;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
    
    for (int i = idx + ind + 1; i < n; i += offsetx) {
        for (int j = idy + ind + 1; j < n; j += offsety) {
            kernel_data[i + j * n] -= kernel_data[i + ind * n] * kernel_data[ind + j * n];
        }
    }
}

__global__ void kernel_calc_l(double *kernel_data, int n, int ind)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    while (idx + ind + 1 < n){
        kernel_data[idx + ind + 1 + ind * n] /= kernel_data[ind + ind * n];
        idx += offsetx;
    }
}

int main()
{
    int n;
    scanf("%d", &n);

    //input matrix
    double *input_data = (double*) malloc(n * n * sizeof(double));
    if (input_data == NULL)
        return -1;

    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            scanf("%lf", &input_data[i + j * n]);
        }
    }

    //permutations array
    int *p = (int *) malloc(n * sizeof(int));
    if (p == NULL)
        return -1;
    
    for (int i = 0; i < n; ++i)
        p[i] = i;

    double *kernel_data;
    CSC(cudaMalloc(&kernel_data, n * n * sizeof(double)));
	CSC(cudaMemcpy(kernel_data, input_data, n * n * sizeof(double), cudaMemcpyHostToDevice));

    comparator comp;
    for (int i = 0; i < n - 1; ++i){
        thrust::device_ptr<double> p_arr = thrust::device_pointer_cast(kernel_data + i * n);
        thrust::device_ptr<double> p_max_ind = thrust::max_element(p_arr + i, p_arr + n, comp);
        int max_ind= p_max_ind - p_arr;
        if (i != max_ind){
            p[i] = max_ind;
            kernel_swap_rows<<<128, 128>>>(kernel_data, n, i, max_ind);
            CSC(cudaGetLastError());
        }
        kernel_calc_l<<<128, 128>>>(kernel_data, n, i);
        CSC(cudaGetLastError());
        kernel_gauss<<<dim3(32, 32), dim3(32, 32)>>>(kernel_data, n, i);
        CSC(cudaGetLastError());
    }

    CSC(cudaMemcpy(input_data, kernel_data, n * n * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            printf("%.10e ", input_data[i + j * n]);
        }
        printf("\n");
    }

    for (int i = 0; i < n; ++i)
        printf("%d ", p[i]);
    printf("\n");

    CSC(cudaFree(kernel_data));
    free(input_data);
    free(p);
    return 0;
}