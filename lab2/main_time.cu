#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CSC(call)  									\
do {											\
	cudaError_t res = call;							\
	if (res != cudaSuccess) {							\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								\
	}										\
} while(0)

#define MIN(a,b) ((a) < (b) ? (a) : (b))

texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void kernel(uchar4 *out, int w, int h) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	uchar4 p1, p2, p3, p4;
	for(int y = idy; y < h; y += offsety){
		for(int x = idx; x < w; x += offsetx) {
			p1 = tex2D(tex, x, y);
			p2 = tex2D(tex, x + 1, y + 1);
			p3 = tex2D(tex, x + 1, y);
			p4 = tex2D(tex, x, y + 1);
			float Y1 = 0.299 * p1.x + 0.587 * p1.y + 0.114 * p1.z;
			float Y2 = 0.299 * p2.x + 0.587 * p2.y + 0.114 * p2.z;
			float Y3 = 0.299 * p3.x + 0.587 * p3.y + 0.114 * p3.z;
			float Y4 = 0.299 * p4.x + 0.587 * p4.y + 0.114 * p4.z;
			float gx = Y2 - Y1;
			float gy = Y4 - Y3;
			int g = MIN(255, sqrt(gx * gx + gy * gy));
			out[y * w + x] = make_uchar4(g, g, g, p1.w);
		}
	}
}

int main() {
	int w, h;
	char in[255], out[255];
	scanf("%s", in);
	scanf("%s", out);
	FILE *fp = fopen(in, "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, w, h));

	CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

	tex.normalized = false;
	tex.filterMode = cudaFilterModePoint;	
	tex.channelDesc = ch;
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;

	CSC(cudaBindTextureToArray(tex, arr, ch));

	uchar4 *dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

	cudaEvent_t start, stop;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));
	CSC(cudaEventRecord(start));

	kernel<<< dim3(32, 32), dim3(32, 32)>>> (dev_out, w, h);
    CSC(cudaDeviceSynchronize());
	CSC(cudaGetLastError());
	
	CSC(cudaEventRecord(stop));
	CSC(cudaEventSynchronize(stop));
	float t;
	CSC(cudaEventElapsedTime(&t, start, stop));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(stop));

	printf("time = %f ms\n", t);

	CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
	CSC(cudaUnbindTexture(tex));
	CSC(cudaFreeArray(arr));
	CSC(cudaFree(dev_out));

	fp = fopen(out, "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	free(data);
	return 0;
}
