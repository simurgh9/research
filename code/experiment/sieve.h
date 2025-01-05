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
#define THREADS 100
#define DENSITY 0.01
#define num double
#define norm unsigned long long
#define SIZE SCALE * DIM
#define INFLATED MAX(SIZE, (int)(ETA * (SIZE - 1) * SIZE / 2))

#define ETA 0.01
// #define SCALE 25

// #define DIM 2
// #define PATH "../../saved/personal/col2_30.csv"

// #define DIM 6
// #define PATH "../../saved/personal/col6_3.csv"

// #define DIM 40
// #define PATH "../../saved/challenge/col40_1740.csv"
// #define PATH "../../saved/personal/col40_237413_212828.csv"
// #define PATH "../../saved/personal/col40_237413_212828.hermite.csv"
// #define PATH "../../saved/personal/exp40_4196653682643_3762083877372.hermite.csv"

// #define SCALE 165
// #define DIM 60
// #define PATH "../../saved/challenge/col60_2101.csv"
// #define PATH "../../saved/personal/col60_1233755_1124757.csv"
// #define PATH "../../saved/personal/col60_1233755_1124757.hermite.csv"

// #define SCALE 714
// #define DIM 70
// #define PATH "../../saved/challenge/col70_2254.csv"
// #define PATH "../../saved/personal/col70_2160237_1979552.csv"
// #define PATH "../../saved/personal/col70_2160237_1979552.hermite.csv"

#define SCALE 10000
#define DIM 100
#define PATH "../../saved/challenge/col100_2667.csv"

int count = 0;
num (*P1)[DIM], (*P2)[DIM], (*P3)[DIM];
norm *norms1, *norms2, *norms3;
int *order1, *order2;
double timestamp = -1;
char underpop_warn[256];
pthread_mutex_t mutie = PTHREAD_MUTEX_INITIALIZER;

void basis_t(char [], num (*)[], norm *); // reads columns into rows 

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
