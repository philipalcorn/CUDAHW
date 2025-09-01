// Name: Phil Alcorn
// nvcc HW3.cu -o temp
/*
 What to do:
 This is the solution to HW2. It works well for adding vectors using a
 single block.
 But why use just one block?
 We have thousands of CUDA cores, so we should use many blocks to keep the 
 SMs (Streaming Multiprocessors) on the GPU busy.

 Extend this code so that, given a block size, it will set the grid size to 
 handle "almost" any vector addition.
 I say "almost" because there is a limit to how many blocks you can use, 
 but this number is very large. 


 We will address this limitation in the next HW.

 Hard-code the block size to be 256.

 Also add cuda error checking into the code.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 11503 // Length of the vector
#define cudaErrCheck cudaErrorCheck(__FILE__, __LINE__);

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.00000001;

// Function prototypes
void setUpDevices();
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float, float, float, int);
int  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();
void cudaErrorCheck(const char* file, int line);
// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = 256;
	BlockSize.y = 1;
	BlockSize.z = 1;

	
	// The following is just a way to say floor (N/BlockSize.x)
	// Or in otherwords, make the gird the minumum size necessary 
	// to have enough threads to handle the entire operation.
	
	// if N = 100, BS = 10: GS = 99/10 + 1 = 10 (100 threads)
	// if N = 101, BS = 10: GS = 100/10 +1 = 11 (110 threads)
	GridSize.x = (N-1)/BlockSize.x +1; 
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
	
	// Device "GPU" Memory
	// We are trying to store an address. To ensure we store the 
	// address in the right pointer, we need to pass the pointer by reference.
	// so we pass the address of the pointer in which we will store the 
	// GPU pointer. Otherwise the GPU address goes into a copy of the pointer 
	// and the pointer we want the address to be in is left unchanged. 
	//
	// It's kind of a mouthful. 
	//
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaMalloc(&C_GPU,N*sizeof(float));

}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i; // 0, 1, 2, 3... 
		B_CPU[i] = (float)(2*i); // 0, 2, 4....
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		c[id] = a[id] + b[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	int id = threadIdx.x;
	
	while(id < n)
	{
		c[id] = a[id] + b[id];
		id += blockDim.x;
	}
}

// Checking to see if anything went wrong in the vector addition.
int check(float *c, int n)
{
	double sum = 0.0;
	double m = n-1; // Needed the -1 because we start at 0.
	
	for(int id = 0; id < n; id++)
	{ 
		sum += c[id];
	}
	
	if(abs(sum - 3.0*(m*(m+1))/2.0) < Tolerance) 
	{
		return(1);
	}
	else 
	{
		return(0);
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
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	cudaFree(A_GPU); 
	cudaFree(B_GPU); 
	cudaFree(C_GPU);
}

void cudaErrorCheck(const char* file, int line)
{
	cudaError_t err;

	err = cudaGetLastError();

	if (err != cudaSuccess) 
	{
		printf("\nCUDA ERROR: message = %s, File = %s, Line = %d\n", 
				cudaGetErrorString(err), file, line);
		exit(0);
	}
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	
	// Setting up the GPU
	setUpDevices();
	cudaErrCheck
	
	// Allocating the memory you will need.
	allocateMemory();
	cudaErrCheck

	// Putting values in the vectors.
	innitialize();
	cudaErrCheck	

	// Adding on the CPU
	gettimeofday(&start, NULL);
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Zeroing out the C_CPU vector just to be safe because right now it 
	// has the correct answer in it.
	for(int id = 0; id < N; id++)
	{ 
		C_CPU[id] = 0.0;
	}
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrCheck
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrCheck
	
	addVectorsGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU ,C_GPU, N);
	cudaErrCheck
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrCheck

	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrCheck
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N) == 0)
	{	
		printf("\n\n Something went wrong in the GPU vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	cudaErrCheck
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n\n");
	
	return(0);
}

