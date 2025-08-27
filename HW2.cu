// Name:
// Vector addition on the GPU, with one block
// To compile: nvcc HW2.cu -o temp
/*
 What to do:
 This code adds the vectors on the GPU.
 Man, that was easy!

 1. First, just add cuda to the word malloc to get cudaMalloc and use 
 it to allocate memory on the GPU.
 Okay, you had to use an & instead of float*, but come on, that was no 
 big deal.

 2. Use cudaMemcpyAsync to copy your CPU memory holding your 
 vectors to the GPU.

 3. Now for the important stuff we've all been waiting for: the GPU 
 "CUDA kernel" that does the work on thousands of CUDA cores all at the
 same time!!!!!!!!  Wait, all you have to do is remove the for loop?
 Dude, that was too simple! I want my money back! 
 Be patient, it gets a little harder, but remember, I told you CUDA was 
 simple.
 
 4. call cudaDeviceSynchronize. Sync up the CPU and the GPU. I'll 
 expaned on this in to story at the end of 5 below.
 
 5. Use cudaMemcpyAsync again to copy your GPU memory back to the CPU.
 Be careful with cudaMemcpyAsync. Make sure you pay attention to the last 
 argument you pass in the call. Also, note that it says "Async" at the end. 
 That means the CPU tells the GPU to do the copy but doesn't wait around for 
 it to finish.

	CPU: 	"Dude, here is your data to work on and don't bother me. 
	It's 'Async'—I’ve got to get back to watching this cool 
	
	TikTok video of a guy smashing watermelons with his face."

	GPU: 	"Whatever, dude. I'll crunch your data when I get around to it. 
	It's 'Async'."

	CPU: 	"Just make sure you get it crunched before I send the results out."

	GPU: 	"Well, maybe you'd better check with me and wait until I'm 
	done before you start publishing results. That means use 
	cudaDeviceSynchronize!"

	CPU: 	"Da."

	GPU: 	"I might be all tied up watching a TikTok video of a guy 
	eating hotdogs with his hands tied behind his back... underwater."

	GPU: 	thought to self: "It must be nice being a CPU, living 
	in the administration zone where time and logic don't apply. 
	Sitting in meetings all day coming up with work for me to do!"

 6. Use cudaFree instead of free.
 
 What you need to do:

 The code below runs for a vector of length 500.
 Modify it so that it runs for a vector of length 1000 and check your result.
 Then, set the vector size to 1500 and check your result again. 
 This is the code you will turn in.
 
 Remember, you can only use one block!!!
 Don’t cry. I know you played with a basket full of blocks when you were a kid.
 I’ll let you play with over 60,000 blocks in the future—you’ll just have to wait.

 Be prepared to explain what you did to make this work and why it works.
 NOTE: Good code should work for any vector length!
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 1500 // Length of the vector


// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
							  //
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid

float Tolerance = 0.01;


// Function prototypes
void setUpDevices();
void allocateMemory();
void innitialize();

void addVectorsCPU(float*, float*, float*, int);

__global__ void addVectorsGPU(float, float, float, int);

bool check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();


// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	
	//Max threads per block is 1024
	BlockSize.x = 1024;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1;
	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	// Allocate the space required for N floats. 
	// Then cast to float* type.
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	//Allocate the space required for N floats on
	//the GPU. 
	//
	//We need the & in &A_GPU becuase otherwise we 
	//are passing a copy of the pointer. We need to use the &
	//becuase otherwise we are storing the malloc'd data at the location
	//of A_GPU. 
	//
	//That is, cudaMalloc gives us an address. We need to put that address 
	//in a box. A_GPU is our box. to put the address in the box,
	//we need the location of the box, which is what &_AGPU gives us. 
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaMalloc(&C_GPU,N*sizeof(float));
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		//fill A_CPU with 1, 2, 3...
		A_CPU[i] = (float)i;	
		//fill B_CPU with 2, 4, 6...
		B_CPU[i] = (float)(2*i);
	}
}


// Adding vectors a and b on the CPU then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		//sequentiall store a+b in c
		c[id] = a[id] + b[id];
	}
}


// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	//threadIdx is one of the built in variables of CUDA. 
	//It is a dim3 type. 
	int id = threadIdx.x;

	//Each thread performs an addition based on their own index.
	//This can be problematic as we are limited by how many threads 
	//we are allowd per block. 
	//
	//So this line by itself doesn't work:
	//c[id] = a[id] + b[id];
	//
	// We need iterate through the kernels somehow. 
	for (int i = id; i<N; i+=blockDim.x) 
	{
		// each thread runs this individually. So thread 1000
		// will run once successfully, add the blocksize, 
		// realize it's greater than N, and break the loop
		// without doing anything. This prevents any sort of 
		// memory access errors. 
		c[i] = a[i] + b[i];

		// by adding the block size in the for loop, 
		// threads 0-1023 get mapped to indices 1024-2047 after 
		// the first iteration. 
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

	// Get the total of the matrix c. If it's different than the 
	// what we know the total should be, there's a error. 
	myAnswer = 0.0;
	for(id = 0; id < n; id++)
	{ 
		myAnswer += c[id];
	}
	
	// Fancy math formula. It's Gauss's method multiplied 
	// by three. (1 for matrix a, and 2 for matrix b since each 
	// element is double the size of matrix a)
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
	// tv_sec equals the number of seconds past the Unix epoch 01/01/1970
	// tv_usec equals the number of microseconds past the current second.
	
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

int main()
{
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
	
	// Copy Memory from CPU to GPU		
	// cudaMemcpyAsync( destination, source, size, direction );
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	
	//Asynchronous simply means non-blocking. We want to allow the CPU (host)
	//to keep working while the transfer is in progress. If you have reached 
	//a point where you need to make sure the transfer is complete, you need
//to synchronize with cudaDeviceSynchronize(). 

	// Devices in the same CUDA stream are synchronized by defult. so the GPU
	// will wait until the memory transfer is complete before calling this 
	// function. 
	addVectorsGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU ,C_GPU, N);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	// This needs to happen because although the GPU is part of its stream, 
	// the CPU is not in a CUDA stream at all. 
	cudaDeviceSynchronize();
	
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
		/*
		 * If the GPU was actually slower than the CPU, it is likely 
		 * that it's becuase you're only using one block. In this case,
		 * kernal initialization and data transfer may 
		 * slow down your GPU and make it slower than your CPU.
		*/
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}

