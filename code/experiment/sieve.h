// -*- compile-command: "gcc -Wpedantic -Ofast -lblas sieve.c -o sieve.o && ./sieve.o" -*-
#define ACCELERATE_NEW_LAPACK 1

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memcpy
#include <time.h>
// #include <cblas.h>

#define ll double
#define THREADS 17

// #define SEED 0
// #define DIM 100
// #define SIZE 1000000
// #define PATH "../../saved/bases/col100_2667.csv"

// #define SEED 0
// #define DIM 70
// #define SIZE 70000
// #define PATH "../../saved/bases/col70_2254.csv"

// #define SEED 0
// #define DIM 60
// #define SIZE 20000
// #define PATH "../../saved/bases/col60_2101.csv"

// #define SEED 0
// #define DIM 40
// #define SIZE 2600
// #define PATH "../../saved/bases/col40_1740.csv"

#define SEED 5
#define DIM 2
#define SIZE 3
#define PATH "../../saved/bases/col2_30.csv"

typedef struct node {
  ll norm;
  ll v[DIM];
  struct node *prev;
  struct node *next;
} node;

double timestamp = -1;
int real = 0;
node *head, *tail;
node *H[THREADS], *T[THREADS];
pthread_mutex_t mutie = PTHREAD_MUTEX_INITIALIZER;

// helper headers
void init(void);
void sieve(void);
void step(void);
void *job(void *);
void merge(void);
void cross(node *, node *, node **);  // https://www.netlib.org/blas/blasqr.pdf
void initial(ll (*B)[], node **, double);
void random_bits(ll *, int, double);
ll dot(ll *, ll *, int);
void basis(char path[], ll (*B)[]);
void print_matrix(ll (*B)[], int, int);
void print_vector(ll *, int);
double mean(node *);
double seconds(void);
double period(void);
int digits(ll);
int delta_real(int);

// list headers
int place(ll, ll *, node **);
int place_from(ll, ll *, node *, node **);
void before(ll, ll *, node *, node **);
void after(ll, ll *, node *, node **);
void pop(node **);
void squeeze(node **);
void free_list(node **);
void print_norms(node *);
void print_list(node *);
node *new_node(ll, ll *, node *);

// maybe useless
void replace(ll, ll *, node *);
void set(ll, ll *, node *);

