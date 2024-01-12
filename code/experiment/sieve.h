#define ACCELERATE_NEW_LAPACK 1
#define ll double

#define SEED 0
#define DIM 100
#define SIZE 1000000
#define PATH "../../saved/bases/col100_2667.csv"

// #define SEED 0
// #define DIM 70
// #define SIZE 70000
// #define PATH "../../saved/bases/col70_2254.csv"

// #define SEED 0
// #define DIM 60
// #define SIZE 10000
// #define PATH "../../saved/bases/col60_2101.csv"

// #define SEED 0
// #define DIM 40
// #define SIZE 600
// #define PATH "../../saved/bases/col40_1740.csv"

// #define SEED 5
// #define DIM 2
// #define SIZE 3
// #define PATH "../../saved/bases/col2_30.csv"

double timestamp = -1;
int real = 0;

typedef struct node {
  int not_ready;
  ll norm;
  ll v[DIM];
  struct node *prev;
  struct node *next;
} node;

// helper headers
void sieve(node **, node **);
void step(node **, node **);
void my_cblas_cross(node *, node *, node **);  // https://www.netlib.org/blas/blasqr.pdf
void initial(ll (*B)[], node **, double);
void random_bits(ll *, int, double);
ll dot(ll *, ll *, int);
void basis(char path[], ll (*B)[]);
void print_matrix(ll (*B)[], int, int);
void print_vector(ll *, int);
double mean(node *);
double seconds();
double period();
int digits(ll);

// list headers
void trim(node **, node **);
int place_from(ll, ll *, node *, node **);
void before(ll, ll *, node *);
void after(ll, ll *, node *);
void append(ll, ll *, node **);
void pop(node **);
void squeeze(node **);
void free_list(node **);
void print_list(node *);
node *new_node(ll, ll *, node *);

// maybe useless
void cross(node *, node *, node **);
int place(ll, ll *, node **);
void replace(ll, ll *, node *);
void set(ll, ll *, node *);

