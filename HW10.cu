// Name: Phil Alcorn
// Robust Vector Dot product 
// nvcc HW10.cu -o temp
/*
 What to do:
 This code is the solution to HW9. It computes the dot product of vectors of any length and uses shared memory to 
 reduce the number of calls to global memory. However, because blocks can't sync, it must perform the final reduction 
 on the CPU. 
 To make this code a little less complicated on the GPU let do some pregame stuff and use atomic adds.
 1. Make sure the number of threads on a block are a power of 2 so we don't have to see if the fold is going to be
    even. Because if it is not even we had to add the last element to the first reduce the fold by 1 and then fold. 
    If it is not even tell your client what is wrong and exit.
	// DONE
 2. Find the right number of blocks to finish the job. But, it is possible that the grid demention is too big. I know
    it is a large number but it is finite. So use device properties to see if the grid is too big for the machine 
    you are on and while you are at it make sure the blocks are not to big too. Maybe you wrote the code on a new GPU 
    but your client is using an old GPU. Check both and if either is out of bound report it to your client then kindly
    exit the program.
	// DONE 
 3. Always checking to see if you have threads working past your vector is a real pain and adds a bunch of time consumming
    if statments to your GPU code. To get around this findout how much you would have to add to your vector to make it 
    perfectly fit in your block and grid layout and pad it with zeros. Multipying zeros and adding zero do nothing to a 
    dot product. If you were luck on HW8 you kind of did this but you just got lucky because most of the time the GPU sets
    everything to zero at start up. But!!!, you don't want to put code out where you are just lucky soooo do a cudaMemset
    so you know everything is zero. Then copy up the now zero values. 
	// DONE
 4. In HW9 we had to do the final add "reduction' on the CPU because we can't sync block. Use atomic add to get around 
    this and finish the job on the GPU. Also you will have to copy this final value down to the CPU with a cudaMemCopy.
    But!!! We are working with floats and atomics with floats can only be done on GPUs with major compute capability 3 
    or higher. Use device properties to check if this is true. And, while you are at it check to see if you have more
    than 1 GPU and if you do select the best GPU based on compute capablity.
	// DONE
 5. Add any additional bells and whistles to the code that you thing would make the code better and more foolproof.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>
// Defines
const long int N = 1'000'000; // Length of the vector
const int BLOCK_SIZE  = 1024; // Threads in a block

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void initialize();
void dotProductCPU(float*, float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();
int check_if_power_of_two(int number);
void check_size_limits(cudaDeviceProp* prop);
int get_best_gpu();
int next_power_of_2 (unsigned int n);
__device__ int __next_power_of_2 (unsigned int n);

int Padded_Vector_Size;
int main()
{

	// get_best_gpu both returns the GPU # and 
	// exits if we don't have a valid gpu. 
	// Two birds with one stone. 
	cudaSetDevice(get_best_gpu());


	if(N> 1'000'000) printf("\nWarning: this method is very"
			" error prone for large N.\nYou might need to reduce"
			" Your vector size.\n\n");
	// Preliminary Checks
	if (!check_if_power_of_two(BLOCK_SIZE)) 
	{
		printf("Your block size needs to be a power of two. Please fix and restart.\n");
		exit(1);
	}

	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	
	// Setting up the GPU
	setUpDevices();
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	check_size_limits(&prop);
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	initialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	printf("CPU Value: %f\n", DotCPU);
	timeCPU = elaspedTime(start, end);
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// blocksize.x is already a power of 2 but I'm leaving the funciton in for 
	// verbosity
	dotProductGPU<<<GridSize,BlockSize, next_power_of_2(BlockSize.x)*sizeof(float)>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	DotGPU = C_CPU[0];
	/*
	for(int i = 0; i < N; i += BlockSize.x)
	{
		DotGPU += C_CPU[i]; // C_GPU was copied into C_CPU. 
	}
	*/

	gettimeofday(&end, NULL);

	printf("GPU Value: %f\n", DotCPU);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\nSomething went wrong in the GPU dot product.\n");
	}
	else
	{
		printf("\n\nYou did a dot product correctly on the GPU");
		printf("\nThe time it took on the CPU was %ld microseconds", timeCPU);
		printf("\nThe time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	cleanUp();	
	
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
		printf("\nCUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(1);
	}
}

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N - 1)/BlockSize.x + 1; // This gives us the correct number of blocks.
	GridSize.y = 1;
	GridSize.z = 1;

	// Gives the total number of threads available.
	// This allows us to pad the vector with zeroes 
	// so that it completely fills the space, ensureing we don't 
	// access any undefined memory when we add the final block
	Padded_Vector_Size = BlockSize.x * GridSize.x;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));

	int padded_size=Padded_Vector_Size*sizeof(float);

	// Device "GPU" Memory
	cudaMalloc(&A_GPU,padded_size);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,padded_size);
	cudaErrorCheck(__FILE__, __LINE__);

	// C_GPU no longer needs to be big
	cudaMalloc(&C_GPU,sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Fill the now slightly biffer 
	cudaMemset(A_GPU, 0, padded_size);
	cudaMemset(B_GPU, 0, padded_size);
}

// Loading values into the vectors that we will add.
void initialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(3*i);
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

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{
	int lid = threadIdx.x;
	int gid = threadIdx.x + blockDim.x*blockIdx.x;
	// Sets the shared memory size to be a power of two
	__shared__ float temp[BLOCK_SIZE];
	
	temp[lid] = (a[gid] * b[gid]);

	// This code should only need to execute on the last block
	// It pads the array with zeros. This should have already been done
	// with the cuda memset but it's one operation to double check.
	if(gid >= n) 
	{
		temp[lid] = 0;
	}
	__syncthreads();

	// blockDim.x is already a power of 2
	// but I'm leaving this in for verbosity's sake
	int len = __next_power_of_2(blockDim.x); 	
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
		atomicAdd(c, temp[0]); 
	}
	//c[blockDim.x*blockIdx.x] = temp[0];
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\npercent error = %lf\n", percentError);
	
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
	
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

int check_if_power_of_two(int number) 
{
	for (int i = 2; i < number; i*=2) 
	{
		if (number % i != 0) return 0;
	}
	return 1;
}

void check_size_limits(cudaDeviceProp* prop) 
{
	if (BlockSize.x > prop->maxThreadsDim[0] || BlockSize.y > prop->maxThreadsDim[1] || BlockSize.z > prop->maxThreadsDim[2] ) 
	{
		printf("You have too many threads per block. Please reduce your thread count"
				" and try again.\n");
		exit (1);
	}

	if (GridSize.x > prop->maxGridSize[0] || GridSize.y > prop->maxGridSize[1] || GridSize.z > prop->maxGridSize[2] ) 
	{
		printf("Your grid size is too large. Please reduce the grid size"
				" and try again.\n");
		exit (1);
	}
}

int get_best_gpu() 
{
	cudaDeviceProp p;	
	int device_count;
	cudaGetDeviceCount(&device_count);
	
	int best_device =-1;
	int best_major, best_minor, best_mp_count = -1;
	int has_valid_gpu = 0;
	  
	for (int i = 0; i < device_count; i++) 
	{
		cudaGetDeviceProperties(&p, i);

		printf("\nDevice %d: %s (CC %d.%d, %d SMs, %.1f GB global mem)\n",
				i, p.name, p.major, p.minor, p.multiProcessorCount,
               (double)p.totalGlobalMem / (1024 * 1024 * 1024));
		
		if(p.major >=3)
		{
			has_valid_gpu = 1;
			if (p.major > best_major || // If the major version is better
				(p.major == best_major && p.minor > best_minor) || // minor is better
				(p.major == best_major && p.minor == best_minor && 
				 p.multiProcessorCount > best_mp_count))// more smps 
			{
				best_device = i;
				best_major=p.major;
				best_minor=p.minor;
				best_mp_count=p.multiProcessorCount;
			}
		}
	}
	printf("\n\n");
	if (!has_valid_gpu) 
	{
		printf("You need a GPU with version 3.0 to run this code.");
		exit(1);
	}
		
	return best_device;
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
