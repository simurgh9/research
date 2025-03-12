#ifdef __APPLE__ 
  #define ACCELERATE_NEW_LAPACK 1
  #include <Accelerate/Accelerate.h>
#else
  #include <cblas.h>
#endif

#ifndef MAX
  #define MAX(a, b) (((a)>(b))?(a):(b))
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memcpy
#include <time.h>
#include <limits.h>
#include <pthread.h>
#include "problems.h"

#define SEED 0
#define THREADS 70
#define DENSITY 0.01
#define ETA 0.01 // inflation factor
#define num double
#define norm unsigned long long
#define SIZE SCALES[DIM] * DIM
#define INFLATED MAX(SIZE, (int)(ETA * (SIZE - 1) * SIZE / 2))

int count = 0;
int SCALES[200] = {0};
num (*P1)[DIM], (*P2)[DIM], (*P3)[DIM];
norm *norms1, *norms2, *norms3;
int *order1, *order2;
double timestamp = -1;
char underpop_warn[256];
pthread_mutex_t mutie = PTHREAD_MUTEX_INITIALIZER;

void basis_t(char [], num (*)[], norm *); // reads columns into rows 

void set_scales(void);
norm initial(num (*)[], norm *, double);
void *sieve(void *arg);
void cross(int, int);
void selection(void);

double mean(void);

void print_matrix(num (*)[], int, int, int order[]);
void print_vector(double *, int, int order[]);

int digits(num);
void swap(void **, void **);
double period(void);
double seconds(void);
int compare1(const void *, const void *);
int compare2(const void *, const void *);


// Local Variables:
// compile-command: "gcc -Wpedantic -Ofast -lblas sieve.c -o sieve.o && \
// ./sieve.o"
// End:
