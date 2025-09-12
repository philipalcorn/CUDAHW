// Name: Phil Alcor
// Simple Julia CPU.
// nvcc HW6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
*/



// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define CHECK() cudaErrorCheck(__FILE__, __LINE__)

//globals- generally bad, but we need this one for GLUT
float* pixels_CPU;

// Constants
// const is better than define as it forces type checking
const float X_LOWER_BOUND = -1.75; 
const float X_UPPER_BOUND = 1.75;
const float Y_LOWER_BOUND = -1.75;
const float Y_UPPER_BOUND =  1.75;

const float A =  -0.824;	//Real part of C
const float B  = -0.1711;	//Imaginary part of C Global variables unsigned int 

const int MAX_MAGNITUDE = 100; // Considered escaped if the value is larger than this
const int MAX_ITERATIONS = 1000; // considered not escaped if you make it this far
						  //
// 32 is square root of 1024, so we process a 1024px^2 area at a time
const int THREAD_WIDTH = 32;
const int THREAD_HEIGHT = 32;

const int WIDTH = 2048;
const int HEIGHT = 2048;


// Function prototypes
void cudaErrorCheck(const char*, int);
void checkAllocation(void* ptr);

void setUpDevices(dim3* blockSize, dim3* gridSize);
void checkSetUpDevices(dim3* b, dim3* g);

// input either x or y pixel coordinates and get corresponding number
// based on size of screen
__device__ float getNumFromPX(int position_px, 
							  int length_dim, 
							  float real_min, 
							  float real_max);
__global__ void checkGetNumFromPX(); 

// x, y, channel (0 = red, 1 = green, 2 = blue);
__device__ int getPixelIndex(int x, int y, int channel);
__global__ void checkGetPixelIndex();

__device__ float getMagnitude(float x, float y);
__global__ void checkGetMagnitude();
//used to progress each iteration of the fractal algorithm
__device__ void step(float* re, float* im, float A, float B);

__global__ void checkEscapeGPU(float* pixel_array);
void display();


int main(int argc, char** argv)
{
	dim3 blockSize;
	dim3 gridSize;

	// allocate space for our RGB values
	pixels_CPU = (float*)malloc(WIDTH*HEIGHT*3*sizeof(float)); 
	checkAllocation(pixels_CPU);

	// We can fill this array on the GPU and then transfer it to the CPU to display
	float* pixels_GPU;
	cudaMalloc(&pixels_GPU, WIDTH*HEIGHT*3*sizeof(float));
	CHECK();

	setUpDevices(&blockSize, &gridSize);
	CHECK();

	checkEscapeGPU<<<gridSize, blockSize>>>(pixels_GPU);
	CHECK();
	cudaDeviceSynchronize();
	CHECK();

	
	cudaMemcpy(pixels_CPU, pixels_GPU,WIDTH*HEIGHT*3*sizeof(float), cudaMemcpyDeviceToHost);
	CHECK();

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Fractals--Man--Fractals");
	glutDisplayFunc(display);
	glutMainLoop();


	//Here are a variety of checks to run to verify functionality
	
	/*
	 
	//Checking our block size setup
	checkSetUpDevices(&blockSize, &gridSize);
	CHECK();

	// Checking our conversion from px coordinate to real number value 
	checkGetNumFromPX<<<1,1>>>(); 
	CHECK();
	cudaDeviceSynchronize(); // need to call this or printf's might not be sent to host
	CHECK();
	
	// Checking our pixel indexing funciton:
	checkGetPixelIndex<<<1,1>>>(); 
	CHECK();
	cudaDeviceSynchronize();
	CHECK();
	
	// Checking the magnitude function:
	checkGetMagnitude<<<1,1>>>(); 
	CHECK();
	cudaDeviceSynchronize();
	CHECK();
	
	*/

	
   	
	
	//Do all the processing on the GPU
	
	// destination, source, direction)
	// display (pixels_CPU);
	
}


void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, \
				File = %s, Line = %d\n", cudaGetErrorString(error), 
				file, line);
		exit(0);
	}
}


void checkAllocation(void* ptr) 
{
	if (!ptr) 
	{
		printf("Memory Allocation failed.\n");
		exit(1);
	}
}


void setUpDevices(dim3* blockSize, dim3* gridSize) 
{
	// One thread per pixel in a 1024x1024 grid, anything larger 
	// needs more blocks 
	blockSize->x = THREAD_WIDTH;
	blockSize->y = THREAD_HEIGHT;
	blockSize->z = 1;

	//The minimum number of blocks to hold the pixels in the x and y directions	
	gridSize->x = (WIDTH-1)/blockSize->x + 1; 
	gridSize->y = (HEIGHT-1)/blockSize->y + 1; 
	gridSize->z = 1; // only two dimensions
}

void checkSetUpDevices(dim3* b, dim3* g) 
{
		
	printf("---Checking Device Setup---\n");
	printf("Block Size: %d, %d, %d\n"
			"Grid Size: %d, %d, %d\n",
			b->x, b->y, b->z,
			g->x, g->y, g->z);
	printf("\n");
}

// This function is designed to work with both real an imaginary. 
// Pass the pixel position (ie, the 0th pixel out of 1023, or maybe the 432nd
// pixel), the length of that dimension (maybe the width and height are 
// different values), and the max and min values on the real number line 
// (ie, maybe you are mapping the pixels from -2 to 2 on the real number line)
//
//
// How spacing between each pixel when mapped to real (or imaginary)
// number line: 
// float px_space = (real_max - real_min) / length_dim; 
//
// Designed to work with array values as inputs, so a 1024 pixel dimention
// takes inputs from 0-1024 for that mapping. 
//
// the length_dim-1 makes sure that the end pixel takes on the exact value 
// of the upper bound.
__device__ float getNumFromPX(int position_px, 
							 int length_dim, 
							 float real_min, 
							 float real_max)
{
	return real_min + (position_px * (real_max - real_min)/(length_dim-1));
} 


__global__ void checkGetNumFromPX() 
{
	float temp1 = getNumFromPX(0,	1024,	 0,	1);
	float temp2 = getNumFromPX(1023,	1024,	 0,	1);
	float temp3 = getNumFromPX(0,	4096,	-2,	1);
	float temp4 = getNumFromPX(4096,	4096,	-2,	1);
	printf("---Checking PX Coordinate Mapping---\n");
	printf("X: mapping from 0 to -1, 1024 pixels wide\n"
			"Pixel pos 0: %f\n" 
			"Pixel pos 1023: %f\n", 
			temp1, temp2);
	printf("Y: mapping from -2 to 1, 4096 pixels tall\n"
			"Pixel pos 0: %f\n"
			"Pixel pos 1023: %f\n", 
			temp3, temp4);
	printf(" (+/- some floating point error)\n");
	printf("\n");
}


__device__ int getPixelIndex(int x, int y, int channel) 
{
	
	return (y * WIDTH + x) * 3 + channel;
}


__global__ void checkGetPixelIndex()
{
	printf("---Checking PX Index Conversion---\n");
	printf("Pixel (0, 0, Blue(2))\n");
	printf("Index: %d\n", getPixelIndex(0,0,2));
	printf("Pixel (100, 2, Green(1))\n");
	printf("Index: %d\n", getPixelIndex(100,2,1));
	printf("If width is 1024x1024, this coordinate should be: \n\n"
			 "\t(y-coord*WIDTH) *3 + x-coord*3 + color_offset \n"
			 "or, \n"
			 "\t2*1024*3 + 100*3 + 1 \n" 
			 "\twhich equals 6,445\n");
	printf("\n");
}


__device__ float getMagnitude(float x, float y) 
{
	return sqrt(x*x + y*y);
}


__global__ void checkGetMagnitude() 
{
	printf("Mag of (3,4): %f\n", getMagnitude(3,4));
}


//Note: This function modifies the original input.
__device__ void step(float* re, float* im, float A, float B) 
{
	float re_initial = *re;

	*re = re_initial*re_initial - (*im)*(*im) + A;
	*im = 2 * re_initial * (*im) + B; 
}


__global__ void checkEscapeGPU(float* pixel_array) 
{
	// Pixel Coordinates come from the initial index
	int pixel_x = blockIdx.x*blockDim.x + threadIdx.x;
	int pixel_y = blockIdx.y*blockDim.y + threadIdx.y;
		
	//we can use the "or equal to" becuase of 0 indexing
	if(pixel_x >= WIDTH || pixel_y >= HEIGHT) return;

	float real_component;
	float imaginary_component;

	//we want to color everything red
	int color_index = getPixelIndex(pixel_x, pixel_y, 0);
	
	real_component = getNumFromPX(pixel_x, WIDTH, X_LOWER_BOUND, X_UPPER_BOUND);
	imaginary_component = getNumFromPX(pixel_y, HEIGHT, Y_LOWER_BOUND, Y_UPPER_BOUND);

	int count =0;
	while (count< MAX_ITERATIONS) 
	{
		if(getMagnitude(real_component, imaginary_component) > MAX_MAGNITUDE)
		{
			// if magnitude exceepds "escape velocity" we color the square red
			pixel_array[color_index]   = 1;	
			pixel_array[color_index+1] = 0;	
			pixel_array[color_index+1] = 0;	
			return;
		}

		step(&real_component, &imaginary_component, A, B);
		count++;
	}

	//If we make it through all the steps, color the square black
	pixel_array[color_index]   = 0;	
	pixel_array[color_index+1] = 0;	
	pixel_array[color_index+1] = 0;	
}

void display(void) 
{ 
	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_FLOAT, pixels_CPU); 
	glFlush(); 
}

