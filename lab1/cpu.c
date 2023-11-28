#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main()
{
    int n;
    scanf("%d", &n);

    float *first_arr = (float *)malloc(sizeof(float) * n);
	float *second_arr = (float *)malloc(sizeof(float) * n);
    float *res_arr = (float *)malloc(sizeof(float) * n);

	for(int i = 0; i < n; i++)
		scanf("%f", &first_arr[i]);
	for(int i = 0; i < n; i++)
		scanf("%f", &second_arr[i]);
	for(int i = 0; i < n; i++)
		res_arr[i] = i;

    clock_t start, end;
    int idx = 0;
    start = clock();
    while (idx < n) 
    {
        if (first_arr[idx] > second_arr[idx])
            res_arr[idx] = first_arr[idx];
        else
            res_arr[idx] = second_arr[idx];
        idx++;
	}

    end = clock();
    double cpu_time = (double) (end - start) / CLOCKS_PER_SEC;

    printf("%f ms\n", cpu_time);

    for(int i = 0; i < n; i++)
		printf("%.10e ", res_arr[i]);
	printf("\n");
}