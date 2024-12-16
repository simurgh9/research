#ifdef __APPLE__ 
  #define ACCELERATE_NEW_LAPACK 1
  #include <Accelerate/Accelerate.h>
#else
  #include <cblas.h>
#endif

#ifndef MAX
  #define MAX(a, b) (((a)>(b))?(a):(b))
#endif

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memcpy
#include <time.h>
#include <limits.h>
#include <pthread.h>

#define SEED 0
#define THREADS 17
// #define THREADS 1
#define DENSITY 0.01
#define num double _Complex
#define norm unsigned long long
#define SIZE SCALE * DIM
#define INFLATED MAX(SIZE, (int)(ETA * (SIZE - 1) * SIZE / 2))

#define ETA 0.01
#define SCALE 300
#define DIM 40

int count = 0;
num (*P1)[DIM], (*P2)[DIM], (*P3)[DIM];
norm *norms1, *norms2, *norms3;
int *order1, *order2;
double timestamp = -1;
char underpop_warn[256];
pthread_mutex_t mutie = PTHREAD_MUTEX_INITIALIZER;

void basis_t(num (*)[], norm *); // reads columns into rows 

norm initial(num (*)[], norm *, double);
void *sieve(void *arg);
void cross(int, int);
void selection(void);

double mean(void);
double uniform(double, double);
num imaginary(double, double);
num imaginary_gaussian(double, double);

void print_matrix(num (*)[], int, int, int order[]);
void print_vector(norm *, int, int order[]);
void print_imag_vector(num *, int);
  
int digits(norm);
void swap(void **, void **);
double period(void);
double seconds(void);
int compare1(const void *, const void *);
int compare2(const void *, const void *);


// Local Variables:
// compile-command: "gcc -Wpedantic -Ofast -lblas isieve.c -o isieve.o && \
// ./isieve.o"
// End:
