/****IMPORTANT*****/
//To Run: ./temp {width} {height}


// Name: Phil Alcorn
// Simple Julia CPU.
// nvcc hw7.cu -o temp -use_fast_math -lglut -lGL
// glut and GL are openGL libraries.

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 But it only runs on a window of 1024X1024.
 Extend it so that it can run on any given window size.
 Also, color it to your liking. I will judge you on your artisct flare. 
 Don't cute off your ear or anything but make Vincent wish he had, had a GPU.
*/

/***** ***** INCLUDES ***** *****/
#include <stdio.h>
#include <GL/glut.h>
#include <sys/time.h>

/***** ***** DEFINES ***** *****/
// Defines
#define CHECK() cudaErrorCheck(__FILE__, __LINE__)

/***** ***** GLOBALS ***** *****/
float* pixels_CPU;
float* pixels_GPU;
dim3 blockSize;
dim3 gridSize;
long long start_time;


/***** ***** ANIMATION PARAMETERS ***** *****/
// Lower numbers give a tigher range of inputs to Julia
const float Z_MOD_VALUE = 0.1; 
const float TIME_SCALE = 0.0007; // Lower values are slower
const float A =  -0.824;	//Real part of C
const float B  = -0.1711;	//Imaginary part of C 
// PI and E are used to ensure we never see the same fractal twice.
const float PI = 3.1415926535897932384626433832795028841971693993751;
const float E = 2.7182818284590452353602874713526624977572470936999;
const float FALLOFF_MIN = 0.7;
const float FALLOFF_MAX = 0.86;
const float FALLOFF_CURVE = 0.7;

/***** ***** CONSTANTS ***** *****/
int WIDTH; // Still "constant" but they are assigned at program start
int HEIGHT; // Still "constant" but they are assigned at program start
const float X_LOWER_BOUND = -1.75; 
const float X_UPPER_BOUND =  1.75;
const float Y_LOWER_BOUND = -1.75;
const float Y_UPPER_BOUND = 1.75;
const int MAX_MAGNITUDE = 20;
const int MAX_ITERATIONS = 200;		// considered not escaped if you make it this far
const int THREAD_WIDTH = 32;		// Square Root of 24
const int THREAD_HEIGHT = 32;

/***** ***** FUNCTION PROTOTYPES ***** *****/
__global__ void colorFractal(	int width, 
								int height, 
								float re, 
								float im, 
								float* pixel_array, 
								float current_time);

__device__ void step(float* re, float* im, float A, float B);

__device__ float getAnimationValue(	float min, 
									float max, 
									float parameter, 
									float time, 
									float timestep);

__device__ float getNumFromPX(int position_px, 
							  int length_dim, 
							  float real_min, 
							  float real_max);

__device__ int getPixelIndex(int x, int y, int channel, int width);

long long get_current_time_ms();

__device__ float getMagnitude(float x, float y);

void display();

void timer(int value);

void setUpDevices(dim3* blockSize, dim3* gridSize);

void cudaErrorCheck(const char*, int);

void checkAllocation(void* ptr);

/***** ***** MAIN ***** *****/
int main(int argc, char** argv)
{
	//argc is arg count, argv is arg vector
	if(argc !=3) 
	{
		printf("Run the program like this: ./<filename> <width> <height>\n");
		exit(1);
	}
	WIDTH = atoi(argv[1]);
	HEIGHT = atoi(argv[2]);

	

	// allocate space for our RGB values
	pixels_CPU = (float*)malloc(WIDTH*HEIGHT*3*sizeof(float)); 
	checkAllocation(pixels_CPU);

	// We can fill this array on the GPU and then transfer it to the CPU to display
	cudaMalloc(&pixels_GPU, WIDTH*HEIGHT*3*sizeof(float));
	CHECK();

	setUpDevices(&blockSize, &gridSize);
	CHECK();
	
	cudaMemcpy(pixels_CPU, pixels_GPU,WIDTH*HEIGHT*3*sizeof(float), cudaMemcpyDeviceToHost);
	CHECK();

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Fractals--Man--Fractals");
	glutDisplayFunc(display);

	start_time = get_current_time_ms();
	glutTimerFunc(16,timer,0);

	glutMainLoop();

	free(pixels_CPU);
	cudaFree(pixels_GPU);
}


/***** ***** FRACTAL GENERATION ***** *****/
// Steps through the the algorithm and determines if we have escaped or not.
// If we have escaped, color the square red. Otherwise color black.
// Fills the data into the single-dimensional pixel_array
__global__ void colorFractal(	int width, 
								int height, 
								float re, 
								float im, 
								float* pixel_array, 
								float current_time) 
{ 
	// This basically alternates our colors by putting them through a 
	// modified sine wave. 
	float A_local = getAnimationValue(re - Z_MOD_VALUE,
								re + Z_MOD_VALUE,
								1,
								current_time,
								TIME_SCALE / (20*E));

	float B_local = getAnimationValue(im - Z_MOD_VALUE,
								im + Z_MOD_VALUE,
								1,
								current_time,
								TIME_SCALE / (20*PI));


	// Orange color: 255, 90, 0
	// Green color: 9, 255, 0
	float red = getAnimationValue(255,9, 4, current_time, TIME_SCALE/4);
	float green = getAnimationValue(90,255, 4, current_time, TIME_SCALE/4);
	float blue =0; // No blue for halloween :(
				   
	float falloff_factor = getAnimationValue(	FALLOFF_MIN,
												FALLOFF_MAX,
												FALLOFF_CURVE, 
												current_time, 
												TIME_SCALE);

	// Pixel Coordinates come from the initial index
	int pixel_x = blockIdx.x*blockDim.x + threadIdx.x;
	int pixel_y = blockIdx.y*blockDim.y + threadIdx.y;
		
	// (We can use the "or equal to" becuase of 0 indexing)
	// Check if IDX is out of bounds of out pixel map
	if(pixel_x >= width || pixel_y >= height) return;

	float real_component;
	float imaginary_component;
	
	//we want to color everything red
	int color_index = getPixelIndex(pixel_x, pixel_y, 0, width);
	real_component = getNumFromPX(pixel_x, width, X_LOWER_BOUND, X_UPPER_BOUND);
	imaginary_component =((float)height/(float)width) *getNumFromPX(	pixel_y,
										height, 
										X_LOWER_BOUND, 
										X_UPPER_BOUND);

	int count = 0;
	float intensity =0;
	while (count< MAX_ITERATIONS) 
	{
		intensity = (MAX_ITERATIONS/150)*pow((float)count/(float)MAX_ITERATIONS, falloff_factor); //Max value is 1
		if(getMagnitude(real_component, imaginary_component) > MAX_MAGNITUDE)
		{
		
			// If we escape, we get a color
			pixel_array[color_index]   = red/255*intensity;	
			pixel_array[color_index+1] = green/255*intensity;	
			pixel_array[color_index+2] = blue/255*intensity;	
			return;
		}

		step(&real_component, &imaginary_component, A_local, B_local);
		count++;
	}

	// If we never escape, we get colored black.
	pixel_array[color_index]   = 0;	
	pixel_array[color_index+1] = 0;	
	pixel_array[color_index+2] = 0;	
}



// Note: This function modifies the original input.
// Applies a single iteration of our fractal algorithm to the input. 
__device__ void step(float* re, float* im, float A, float B) 
{
	float re_initial = *re;

	*re = re_initial*re_initial - (*im)*(*im) + A;
	*im = 2 * re_initial * (*im) + B; 
}


/***** ***** ANIMATION FUNCTIONS ***** *****/
// "lingering" sine wave
// https://math.stackexchange.com/questions/100655/cosine-esque-function-with-flat-peaks-and-valleys
__device__ float getAnimationValue(	float min, 
									float max, 
									float parameter, 
									float time, 
									float timescale) 
{
	float amplitude = (max-min)/2;
	float center = (max+min)/2;

	return amplitude * __sinf(time*timescale) *
		sqrt( 
			(1+pow(parameter,2)) 
			/ 
			(1+pow(parameter*__sinf(time*timescale), 2) 
		))
		+ center;
}


/***** ***** UTILITIES ***** *****/
// Maps a pixel value to a real number. 
// The pixel position is either the x or y coordinate, and then the 
// Other three parameters are relevant to that dimension only (all x or all y)
__device__ float getNumFromPX(int position_px, 
							 int length_dim, 
							 float real_min, 
							 float real_max)
{
	return real_min + (position_px * (real_max - real_min)/(length_dim-1));
}

//Given an X,Y pixel, returns the position in the single dimension RGB array
// (which is three times as long as the corresponding single dimension 
// pixel array)
__device__ int getPixelIndex(int x, int y, int channel, int width) 
{
	
	return (y * width + x) * 3 + channel;
}

long long get_current_time_ms()
{
	struct timeval tv; 
	gettimeofday(&tv, NULL); 
	return (long long)(tv.tv_sec)*1000 + (tv.tv_usec / 1000);
}


__device__ float getMagnitude(float x, float y) 
{
	return sqrt(x*x + y*y);
}


/***** **** GLUT FUNCTIONS ***** *****/
void display(void) 
{ 
	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_FLOAT, pixels_CPU); 
	glutSwapBuffers(); 
}

void timer (int value) 
{
	long long time_elapsed = get_current_time_ms() - start_time;
	colorFractal<<<gridSize,blockSize>>>(WIDTH, HEIGHT, A, B, pixels_GPU, time_elapsed);
	CHECK();
	cudaDeviceSynchronize();

	cudaMemcpy(pixels_CPU, pixels_GPU,
			WIDTH*HEIGHT*3*sizeof(float),
			cudaMemcpyDeviceToHost);
	glutPostRedisplay();

	glutTimerFunc(16,timer,0);
}


/***** **** SETUP ***** *****/
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


/***** **** TESTING ***** *****/
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

