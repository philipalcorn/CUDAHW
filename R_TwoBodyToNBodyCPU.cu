// Name: Phil 
// Two body problem
// nvcc 17TwoBodyToNBodyCPU.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.
*/

/*
 Purpose:
 To learn about Nbody code.
*/
// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Globals
int tdraw = 0;
float current_time = 0.0;

const int XWindowSize = 1000;
const int YWindowSize = 1000;
const int STOP_TIME =	10000.0;
const float DT = 0.0001;
const float GRAVITY = 1 ;
const float MASS = 10.0;
const float DIAMETER = 1.0;
const float SPHERE_PUSH_BACK_STRENGTH =	50.0;
const float PUSH_BACK_REDUCTION = 0.1;
const float DAMP = 0.01;
const float DRAW = 100;
const float LENGTH_OF_BOX =	6.0;
const float MAX_VELOCITY =	1.0;

const int NUMBER_OF_BODIES = 5;

const float XMAX = (LENGTH_OF_BOX/2.0);
const float YMAX = (LENGTH_OF_BOX/2.0);
const float ZMAX = (LENGTH_OF_BOX/2.0);
const float XMIN = -(LENGTH_OF_BOX/2.0);
const float YMIN = -(LENGTH_OF_BOX/2.0);
const float ZMIN = -(LENGTH_OF_BOX/2.0);


// Defining a 3d vector and it's functions
typedef struct 
{
	float x;
	float y;
	float z;
} vec3;

vec3 vec3_add(vec3 a, vec3 b);
vec3 vec3_subtract(vec3 a, vec3 b);
vec3 vec3_multiply(vec3 a, float b);
vec3 vec3_divide(vec3 a, float b);
vec3 vec3_negate(vec3 a);
float vec3_dot(vec3 a, vec3 b);
vec3 vec3_cross(vec3 a, vec3 b);
float vec3_length(vec3 a);
float vec3_length_squared(vec3 a);
vec3 vec3_normalize(vec3 a);

// Defining an N body and it's functions
typedef struct {
    vec3 position;
    vec3 velocity;
    vec3 force;
	int mass;

} Body;

Body bodies[NUMBER_OF_BODIES];

int check_initial_positions(Body* bodies, int length);
void set_initial_conditions(Body* bodies, int bodies_length);
void get_forces(Body* bodies, int bodies_length);
void move_bodies(Body* bodies, int bodies_length, float time);
void check_axis(float* position, float* velocity, float hbl);
void keep_in_box(Body* bodies, int bodies_length);

// Function prototypes
void Drawwirebox();
void draw_picture();
void move_bodies(float);
void nbody(Body* bodies, int bodies_length);
void display(void);
void reshape(int, int);

void handle_key(unsigned char key, int x, int y);

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("2 Body 3D");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(handle_key);
	set_initial_conditions(bodies, NUMBER_OF_BODIES);
	glutMainLoop();
	return 0;
}


void set_initial_conditions(Body* bodies, int bodies_length)
{ 
	time_t t;
	srand((unsigned) time(&t));
	
	//Filling out the values in the body array 
	for (int i = 0; i < bodies_length; i++)
	{
		// Assigning a valid position
		int pos_is_valid = 0; // 0 for invalid 1 for valid
		while (!pos_is_valid) 
		{
			// This format makes sure they are all intially inside the box
			bodies[i].position.x = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			bodies[i].position.y = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			bodies[i].position.z = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			
			// This function loops through the first i-1 bodies to make sure 
			// the ith body doesn't collide with them
			pos_is_valid = check_initial_positions(bodies, i);	
		}

		// Assigning a velocity
		bodies[i].velocity.x = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		bodies[i].velocity.y = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		bodies[i].velocity.z = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;

		
		// Assigning Mass
		bodies[i].mass = MASS;
		
		// Assigning a current force on that body
		bodies[i].force.x = 0;
		bodies[i].force.y = 0; 
		bodies[i].force.z = 0; 
	}
}

void draw_picture(Body* bodies, int bodies_length)
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();
	
	for (int i = 0; i< bodies_length; i++) 
	{
		vec3 pos = bodies[i].position;

		glColor3d(1.0,0.5,1.0);
		glPushMatrix();
		glTranslatef(pos.x, pos.y, pos.z);
		glutSolidSphere(radius,20,20);
		glPopMatrix();
	}

	glutSwapBuffers();
}

void check_axis(float* position, float* velocity, float hbl) 
{
	if (abs(*position) > hbl) 
	{
		float pos_or_neg = (*position > 0) ? 1 : -1;

		// Adjusting the position depends on if it's positive or negative
		*position = pos_or_neg * 2.0f * hbl - *position;

		// If we're out of bounds we flip velocity no matter what
		*velocity = -*velocity;
	}
}

void keep_in_box(Body* bodies, int bodies_length)
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;

	for (int i = 0; i < bodies_length; i++) 
	{
		check_axis(&bodies[i].position.x, &bodies[i].velocity.x, halfBoxLength);
		check_axis(&bodies[i].position.y, &bodies[i].velocity.y, halfBoxLength);
		check_axis(&bodies[i].position.z, &bodies[i].velocity.z, halfBoxLength);
	}
}

vec3 compute_force(Body a, Body b) 
{
	vec3 delta_s = vec3_subtract(b.position, a.position);
	float r_squared = vec3_dot(delta_s, delta_s); // r = radius
	float r = sqrt(r_squared);

	// Avoid division by zero
	// This also allows us to compute the force between the body 
	// and itself, which will help our loops later
	if (r == 0.0f) { return (vec3){0,0,0}; }

	float magnitude = a.mass * b.mass * GRAVITY / r_squared; // universal law
															 //
	// handling collision
	if (r < DIAMETER) 
	{
		vec3 delta_v = vec3_subtract(a.velocity, b.velocity);
		float inout = vec3_dot(delta_s, delta_v);
		float overlap = r-DIAMETER;

		if (inout < 0.0f) 
		{
			magnitude += SPHERE_PUSH_BACK_STRENGTH * overlap;
		} else 
		{
			magnitude += PUSH_BACK_REDUCTION* SPHERE_PUSH_BACK_STRENGTH * overlap;
		}
	}

	// Generates a unit vector in the direction of delta 
	vec3 direction = vec3_divide(delta_s, r); 
	// Scales the unit vector by the magnitude of force
	return vec3_multiply(direction, magnitude);	
}

void get_forces(Body* bodies, int bodies_length)
{
	for (int i = 0; i < bodies_length; i ++) 
	{
		bodies[i].force = (vec3){0,0,0}; // Reset the force first
	}

	for (int i = 0; i < bodies_length-1; i ++) 
	{
		
		for (int j = i+1; j < bodies_length; j++) 
		{
			vec3 f = compute_force(bodies[i],bodies[j]);
			bodies[i].force = vec3_add(bodies[i].force, f);
			bodies[j].force = vec3_subtract(bodies[j].force, f); // Adding the negative to body 2
		}
	}

}

void move_bodies(Body* bodies, int bodies_length)
{
	for (int i = 0; i < bodies_length; i++ ) 
	{
		Body* b = &bodies[i]; // need the pointer this time to update the actual values
		
		// If we're at time 0, only step half of DT
		float actual_dt = (current_time == 0) ? 0.5 * DT : DT; // if we're 
		
		vec3 acceleration = vec3_divide(b->force, b->mass);
		vec3 damp_term = vec3_multiply(b->velocity, DAMP);

		// I just rearranged the formula slightly to make it work properly with
		// the vector functions. I can explain the derivation in class
		b->velocity = vec3_add(b->velocity, vec3_multiply(vec3_subtract(acceleration, damp_term), actual_dt));

		// Velocity is always multiplied by DT
		b->position = vec3_add(b->position, vec3_multiply(b->velocity, DT));
	}
	keep_in_box(bodies, bodies_length);
}

void nbody(Body* bodies, int bodies_length)
{	
	if(current_time >= STOP_TIME) 
	{
		printf("\nDONE\n");	
		return;
	}

	set_initial_conditions(bodies, bodies_length);
	draw_picture(bodies, bodies_length);

	while(current_time < STOP_TIME) 
	{
		get_forces(bodies, bodies_length);
		
		move_bodies(bodies, bodies_length);

		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(bodies, bodies_length); 
			tdraw = 0;
		}

		current_time += DT;
	}
	
}

// GL Functions
void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMAX,YMAX,ZMAX);
		glVertex3f(XMAX,YMAX,ZMIN);	
		glVertex3f(XMAX,YMIN,ZMIN);
		glVertex3f(XMAX,YMIN,ZMAX);
		glVertex3f(XMAX,YMAX,ZMAX);
		
		glVertex3f(XMIN,YMAX,ZMAX);
		
		glVertex3f(XMIN,YMAX,ZMAX);
		glVertex3f(XMIN,YMAX,ZMIN);	
		glVertex3f(XMIN,YMIN,ZMIN);
		glVertex3f(XMIN,YMIN,ZMAX);
		glVertex3f(XMIN,YMAX,ZMAX);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMIN,YMIN,ZMAX);
		glVertex3f(XMAX,YMIN,ZMAX);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMIN,YMIN,ZMIN);
		glVertex3f(XMAX,YMIN,ZMIN);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMIN,YMAX,ZMIN);
		glVertex3f(XMAX,YMAX,ZMIN);		
	glEnd();
	
}


void handle_key(unsigned char key, int x, int y)
{
    if (key == 27)  // 27 = ASCII code for ESC
    {
        exit(0);    // Immediately terminate the program
    }
}
void display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody(bodies, NUMBER_OF_BODIES);
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}



// Vector Functions start here
vec3 vec3_add(vec3 a, vec3 b) 
{
	return (vec3) { a.x+b.x, a.y+b.y, a.z+b.z };
}

vec3 vec3_subtract(vec3 a, vec3 b) 
{
	return (vec3) { a.x-b.x, a.y-b.y, a.z-b.z };
} 

vec3 vec3_multiply(vec3 a, float b) 
{
	return (vec3) { a.x*b, a.y*b, a.z*b };
} 

vec3 vec3_divide(vec3 a, float b) 
{
	return (vec3) { a.x/b, a.y/b, a.z/b };
} 

vec3 vec3_negate(vec3 a) 
{
	return (vec3) {-a.x, -a.y, -a.z};
}

float vec3_dot(vec3 a, vec3 b) 
{
	return (float)( a.x*b.x + a.y*b.y + a.z*b.z );
}

vec3 vec3_cross(vec3 a, vec3 b) 
{
	return (vec3)
	{
		a.y*b.z - a.z*b.y,
		a.z*b.x - a.x*b.z,
		a.x*b.y - a.y*b.x
	};
}

float  vec3_length (vec3 a) 
{
	return (float) sqrt(vec3_dot(a, a));
}

float  vec3_length_squared (vec3 a) 
{
	return  a.x*a.x + a.y*a.y + a.z*a.z ;
}

vec3 vec3_normalize(vec3 a) 
{
	return vec3_divide(a, vec3_length(a));
}

// Body Functions start here
int check_initial_positions(Body* bodies, int length) 
{
	if (length == 0) { return 1; } // If it's the first body, assume it's valid
	
	Body current_body = bodies[length];
	for (int i = 0; i < length; i++) 
	{
		vec3 delta = vec3_subtract(current_body.position, bodies[i].position);
		float r_squared = vec3_dot(delta, delta);
		if (r_squared < DIAMETER*DIAMETER) { return 0; } // avoid the sqrt
	}
	
	// if we've made it this far then all the bodies are valid.
	return 1; 
}


