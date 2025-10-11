// Name:
// Vector Dot product on many block and useing shared memory
// nvcc HW9.cu -o temp
/*
 What to do:
 This code is the solution to HW8. It finds the dot product of vectors that are smaller than the block size.
 Extend this code so that it sets as many blocks as needed for a set thread count and vector length.
 Use shared memory in your blocks to speed up your code.
 You will have to do the final reduction on the CPU.
 Set your thread count to 200 (block size = 200). Set N to different values to check your code.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>

// Defines
const uint64_t  N = 500000; // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float* temp_sums_GPU;
float* temp_sums_CPU;
float DotCPU = 0;
float DotGPU =0;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();
int next_power_of_2(unsigned int n);
__device__ int __next_power_of_2(unsigned int n);


int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	printf("CPU TOTAL: %f\n",C_CPU[0]); 

	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpy(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,
					BlockSize, 
					next_power_of_2(BlockSize.x)*sizeof(float)>>>
					(A_GPU, B_GPU, temp_sums_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpy(temp_sums_CPU, temp_sums_GPU, 
					((N-1)/BlockSize.x + 1 )*sizeof(float), 
					cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);

	float total =0;
	for (int i = 0; i < GridSize.x; i++ ) 
	{
		total += temp_sums_CPU[i];
	}

	DotGPU = total; // C_GPU was copied into C_CPU.
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	printf("GPU TOTAL: %f\n", total);
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
	}
	else
	{
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	cleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}


void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = 200;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N-1)/BlockSize.x + 1;
	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	temp_sums_CPU = (float*)malloc( ((N-1)/BlockSize.x+1) * sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	cudaMalloc(&temp_sums_GPU, ((N-1)/BlockSize.x+1) * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(i*3);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		C_CPU[id] = a[id] * b[id];
	}
	
	for(int id = 1; id < n; id++)
	{ 
		C_CPU[0] += C_CPU[id];
	}
}


__global__ void dotProductGPU(float *a, float *b, float* temp_sums_GPU, int n)
{
	extern	__shared__ float temp[];
	int gid = blockDim.x*blockIdx.x + threadIdx.x;
	int lid = threadIdx.x;
	
	//initialize temp to 0 (We know temp has a size of npo2 (blockDim.x)
	int midpoint = __next_power_of_2(blockDim.x) / 2;
	if (lid<midpoint) 
	{
		temp[lid] = 0;
		temp[lid+midpoint] = 0;
	}
	//if (lid < 56) 
	//{
		//temp[blockDim.x + lid] = 0;
	//}
	
	// Multiplication:	
	// We are setting temp to be the size of the next power of 2 from N
	// However we only have threads going up to size N. So only the first N
	// indices need to be assigned as the rest are padded to 0 for the
	// reduction algorithm.
	
	if (gid < n) 
	{
		temp[lid] = a[gid] * b[gid]; // Doing the multiplication
	}
	__syncthreads();

	// Reduction:
	// Reduction logic 0is local, so it can use the threadIdx.x.
	int len = __next_power_of_2(blockDim.x)  ; // Largest power of 2 to avoid integer division issues
	while (len > 1)
	{
		if (lid< len/2)  { 
			temp[lid] = temp[lid] + temp[lid+len/2];
		}
		__syncthreads();
		len = (len+1) / 2; 
	}

	if (lid == 0) 
	{
		temp_sums_GPU[blockIdx.x] = temp[0];
	}
	
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = fabs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void cleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	free(temp_sums_CPU);
	
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(temp_sums_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}



int next_power_of_2(unsigned int n) 
{
	uint64_t temp = --n;
	temp |= temp >> 1;
	temp |= temp >> 2;
	temp |= temp >> 4;
	temp |= temp >> 8;
	temp |= temp >> 16;
	temp |= temp >> 32;
	temp++;
	return temp;
}
__device__ int __next_power_of_2(unsigned int n)
{
	uint64_t temp = --n;
	temp |= temp >> 1;
	temp |= temp >> 2;
	temp |= temp >> 4;
	temp |= temp >> 8;
	temp |= temp >> 16;
	temp |= temp >> 32;
	temp++;
	return temp;
}

