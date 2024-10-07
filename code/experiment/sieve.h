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

#define SEED 0
#define THREADS 300
#define DENSITY 0.01
#define num double
#define ETA 0.01

// #define BETA 1
// #define DIM 2
// #define PATH "../../saved/bases/col2_30.csv"

// #define BETA 25
// #define DIM 6
// #define PATH "../../saved/bases/col6_3.csv"

// #define BETA 25
// #define DIM 26
// #define PATH "../../saved/personal/general.csv"

// #define BETA 25
// #define DIM 40
// #define PATH "../../saved/bases/col40_1740.csv"

// #define DIM 60
// #define BETA 165
// #define PATH "../../saved/bases/col60_2101.csv"

// #define BETA 714
// #define DIM 70
// #define PATH "../../saved/bases/col70_2254.csv"

#define BETA 5000
#define DIM 100
#define PATH "../../saved/bases/col100_2667.csv"

int SIZE = BETA * DIM;
int INFLATED = MAX(BETA * DIM, (int)(((ETA * (BETA * DIM - 1) * (BETA * DIM)) / 2)));
int count = 0;
num (*P1)[DIM];
num (*P2)[DIM];
num (*P3)[DIM];
num *norms1;
num *norms2;
int *order1;
int *order2;
num *norms3;
double timestamp = -1;
pthread_mutex_t mutie = PTHREAD_MUTEX_INITIALIZER;

void *sieve(void *arg);
void cross(int, int);
void selection(void);
void initial(num (*)[], num *, double);
void basis_t(char [], num (*)[], num *); // reads columns into rows 
int compare1(const void *, const void *);
int compare2(const void *, const void *);
void print_matrix(num (*)[], int, int);
void print_vector(num *, int);
double seconds(void);
int digits(num);
void swap(void **, void **);
double period(void);

// Local Variables:
// compile-command: "gcc -Wpedantic -Ofast -lblas sieve.c -o sieve.o && \
// ./sieve.o"
// End:
