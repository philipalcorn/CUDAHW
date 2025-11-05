// Name: Phil Alcorn
// Vector addition on two GPUs.
// nvcc HW22.cu -o temp
/*
 What to do:
 This code adds two vectors of any length on a GPU.
 Rewriting the Code to Run on Two GPUs:

 1. Check GPU Availability:
    Ensure that you have at least two GPUs available. If not, report the issue and exit the program.

 2. Handle Odd-Length Vector:
    If the vector length is odd, ensure that you select a half N value that does not exclude the last element of the vector.

 3. Send First Half to GPU 0:
    Send the first half of the vector to the first GPU, and perform the operation of adding a to b.

 4. Send Second Half to GPU 1:
    Send the second half of the vector to the second GPU, and again perform the operation of adding a to b.

 5. Return Results to the CPU:
    Once both GPUs have completed their computations, transfer the results back to the CPU and verify that the results are correct.
*/

/*
 Purpose:
 To learn how to use multiple GPUs.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
int N = 11111503; // Length of the vector
// changed to int for type checks

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU_1, *B_GPU_1, *C_GPU_1; //GPU pointers for the first half of the numbers
float *A_GPU_2, *B_GPU_2, *C_GPU_2; //GPU pointers for the secondd half of the numbers 
int length_1;
int length_2; // Lengths of the two separate GPU vectors
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float*, float*, float*, int);
bool  check(float*, int, float);
long elaspedTime(struct timeval, struct timeval);
void CleanUp();

cudaStream_t stream1, stream2;


int main()
{

	int num_gpus;
	cudaGetDeviceCount(&num_gpus);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Checking device count 
	if ( num_gpus != 2 ) 
	{
		printf("This code requires two gpus.\n");
		exit(0);
	}
	
	
	//Set our lengths	
	length_1 = N / 2; 
	length_2 = N-length_1 ;

	
	timeval start, end;
	long timeCPU, timeGPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Zeroing out the C_CPU vector just to be safe because right now it has the correct answer in it.
	for(int id = 0; id < N; id++)
	{ 
		C_CPU[id] = 0.0;
	}
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU 1
	cudaSetDevice(0);
	cudaMemcpyAsync(A_GPU_1, A_CPU, length_1*sizeof(float), cudaMemcpyHostToDevice, stream1);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU_1, B_CPU, length_1*sizeof(float), cudaMemcpyHostToDevice, stream1);
	cudaErrorCheck(__FILE__, __LINE__);

	
	// Copy Memory from CPU to GPU 2
	cudaSetDevice(1);
	cudaMemcpyAsync(A_GPU_2, A_CPU+length_1, length_2*sizeof(float), cudaMemcpyHostToDevice, stream2);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU_2, B_CPU+length_1, length_2*sizeof(float), cudaMemcpyHostToDevice, stream2);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaSetDevice(0);
	addVectorsGPU<<<GridSize,BlockSize, 0, stream1>>>(A_GPU_1, B_GPU_1, C_GPU_1, length_1);
	cudaErrorCheck(__FILE__, __LINE__);

	cudaSetDevice(1);
	addVectorsGPU<<<GridSize,BlockSize, 0, stream2>>>(A_GPU_2, B_GPU_2, C_GPU_2, length_2);
	cudaErrorCheck(__FILE__, __LINE__);

	cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__,__LINE__);
	
	cudaSetDevice(1);
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__,__LINE__);
	
	
	// Copy Memory from GPU to CPU	
	cudaSetDevice(0);
	cudaMemcpyAsync(C_CPU, C_GPU_1, length_1*sizeof(float), cudaMemcpyDeviceToHost, stream1);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaSetDevice(1);
	cudaMemcpyAsync(C_CPU+length_1, C_GPU_2, length_2*sizeof(float), cudaMemcpyDeviceToHost, stream2);
	cudaErrorCheck(__FILE__, __LINE__);
	

	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N, Tolerance) == false)
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
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}



// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
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
	BlockSize.x = 256;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N - 1)/BlockSize.x + 1; // This gives us the correct number of blocks.
	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	cudaMallocHost(&A_CPU, N * sizeof(float));	
	cudaMallocHost(&B_CPU, N * sizeof(float));	
	cudaMallocHost(&C_CPU, N * sizeof(float));	


	// Device "GPU" Memory
	cudaSetDevice(0);
	cudaMalloc(&A_GPU_1,length_1*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU_1,length_1*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU_1,length_1*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaStreamCreate(&stream1);
	
	// Allocate the second device mem
	cudaSetDevice(1);
	cudaMalloc(&A_GPU_2,length_2*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU_2,length_2*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU_2,length_2*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaStreamCreate(&stream2);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
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
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id < n) // Making sure we are not working on memory we do not own.
	{
		c[id] = a[id] + b[id];
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float *c, int n, float tolerence)
{
	int id;
	double myAnswer;
	double trueAnswer;
	double percentError;
	double m = n-1; // Needed the -1 because we start at 0.
	
	myAnswer = 0.0;
	for(id = 0; id < n; id++)
	{ 
		myAnswer += c[id];
	}
	
	trueAnswer = 3.0*(m*(m+1))/2.0;
	
	percentError = abs((myAnswer - trueAnswer)/trueAnswer)*100.0;
	
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
void CleanUp()
{
	// Freeing host "CPU" memory.
	cudaFreeHost(A_CPU); 
	cudaFreeHost(B_CPU); 
	cudaFreeHost(C_CPU);
	
	cudaFree(A_GPU_1); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU_1); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU_1);
	cudaErrorCheck(__FILE__, __LINE__);

	cudaFree(A_GPU_2); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU_2); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU_2);
	cudaErrorCheck(__FILE__, __LINE__);

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

}


