// gcc -Ofast -lblas sieve.c -o sieve.o && ./sieve.o
// gcc -L/opt/homebrew/opt/openblas/lib -I/opt/homebrew/opt/openblas/include -Ofast -lopenblas sieve.c
#include "sieve.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memcpy
#include <time.h>
// #include <Accelerate/Accelerate.h>
#include <cblas.h>

int main(int argc, char *argv[]) {
  period();
  double beg = seconds();
  srand(SEED);
  ll B[DIM][DIM];
  basis(PATH, B);
  node *head = new_node(-1, NULL, NULL);
  node *tail = head;
  initial(B, &tail, 0.01);
  sieve(&head, &tail);
  printf("Total time: %fs\n", seconds() - beg);
  print_vector(tail->v, DIM);
  return 0;
}

// helper implements
void sieve(node **head, node **tail) {
  int i = 0;
  ll unchanged_for = 0;
  ll best = (*tail)->norm;
  char fmt[] = "%4d %8.3f, %8.3f with %5d in %.3fs\n";
  do {
    printf(fmt, i, sqrt(best), mean(*head), real, period());
    step(head, tail);
    if (best == (*tail)->norm)
      unchanged_for++;
    else
      unchanged_for = 0;
    best = (*tail)->norm;
    i++;
  } while (unchanged_for < 10);
}

void step(node **head, node **tail) {
  node *b, *a = *head;
  while ((a = a->next) != NULL && real < 5 * SIZE) {
    if (a->not_ready)
      continue;
    b = a;
    while ((b = b->next) != NULL && real < 5 * SIZE) {
      if (b->not_ready)
        continue;
      my_cblas_cross(a, b, tail);
    }
  }
  trim(head, tail);
}

void my_cblas_cross(node *a, node *b, node **tail) {
  ll t[DIM], tn = 0;
  ll an = a->norm, bn = b->norm;
  double numerator = dot(b->v, a->v, DIM);
  ll m = (ll)round(numerator / bn);
  cblas_dcopy(DIM, a->v, 1, t, 1);
  cblas_daxpy(DIM, -1*m, b->v, 1, t, 1);
  tn = dot(t, t, DIM);
  if (tn < bn)
    place_from(tn, t, b, tail);
  else if (tn < an)
    place_from(tn, t, a, tail);
}

void initial(ll (*B)[DIM], node **tail, double p1) {
  ll cur[DIM], coefs[DIM], norm = 0;
  int i = 0, j = 0;
  for (i = 0; i < DIM; i++) {
    for (j = 0; j < DIM; j++) {
      cur[j] = B[j][i];
      norm += cur[j] * cur[j];
    }
    place(norm, cur, tail);
    norm = 0;
  }
  while (i < SIZE) {
    random_bits(coefs, DIM, p1);
    for (j = 0; j < DIM; j++) {
      cur[j] = dot(coefs, B[j], DIM);
      norm += cur[j] * cur[j];
    }
    i += place(norm, cur, tail);
    norm = 0;
  }
  node *x = *tail;
  do
    x->not_ready = 0;
  while ((x = x->prev) != NULL);
}

void random_bits(ll *fill, int size, double p1) {
  for (int i = 0; i < size; i++)
    fill[i] = (rand() / ((double)RAND_MAX)) < p1;
}

double dot(ll *x, ll *y, int size) {
  return cblas_ddot(size, x, 1, y, 1);
}

void basis(char path[], ll (*B)[DIM]) {
  int i = 0, j = 0, c, sign = 1;
  ll number = 0;
  FILE *fp = fopen(path, "r");
  if (!fp)
    printf("File %s not found.\n", path);
  while ((c = getc(fp)))
    if (c == '-') {
      sign = -1;
    } else if (c == ',') {
      B[i][j++] = sign * number;
      sign = 1;
      number = 0;
    } else if (c == '\n') {
      B[i++][j] = sign * number;
      j = 0;
      sign = 1;
      number = 0;
    } else if (c == EOF) {
      B[i][j] = sign * number;
      break;
    } else if ('0' >= c || c <= '9') {
      number = (number * 10) + (c - '0');
    }
  fclose(fp);
}

void print_matrix(ll (*B)[DIM], int row, int column) {
  int dg = -1, cur = 0, i = 0, j = 0;
  for (i = 0; i < row; i++)
    for (j = 0; j < column; j++)
      if ((cur = digits(B[i][j])) > dg)
        dg = cur;
  dg += 2;
  for (i = 0; i < row; i++) {
    for (j = 0; j < column; j++)
      printf("%*lld", dg, (long long)B[i][j]);
    printf("\n");
  }
}

void print_vector(ll *x, int size) {
  int dg = -1, cur = 0, i = 0;
  for (i = 0; i < size; i++)
    if ((cur = digits(x[i])) > dg)
      dg = cur;
  dg += 2;
  int len = 70 / dg;
  for (int i = 0; i < size; i += len) {
    for (int j = i; j < i + len && j < size; j++) {
      if (j == 0)
        printf("[%*lld", dg - 1, (long long)x[j]);
      else
        printf("%*lld", dg, (long long)x[j]);
    }
    if (i + len < size)
      printf("\n");
    else
      printf("]\n");
  }
}

double mean(node *head) {
  double s = 0, n = 0;
  while ((head = head->next) != NULL) {
    s += sqrt(head->norm);
    n++;
  }
  return s / n;
}

double seconds() {
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return tp.tv_sec + tp.tv_nsec / 1000000000.0;
}

double period() {
  double t = seconds();
  if (timestamp < 0) {
    timestamp = t;
    return -1;
  }
  double ret = t - timestamp;
  timestamp = t;
  return ret;
}

int digits(ll n) {
  if (n == 0)
    return 1;
  return floor(log10(n)) + 1;
}

// list implements
void trim(node **head, node **tail) {
  int n = 1;
  node *x = *tail;
  do
    x->not_ready = 0;
  while ((x = x->prev) != NULL && n++ < SIZE);
  if (n <= SIZE || x == *head) // don't free dummy head
    return;
  (*head)->next->prev = NULL;
  (*head)->next = x->next;
  x->next->prev = *head;
  free_list(&x);
}

int place_from(ll norm, ll v[DIM], node *from, node **tail) {
  node *x = from;
  while (x->next != NULL && norm < x->norm)
    x = x->next;
  if (norm == x->norm || norm == 0)
    return 0;
  if (x->next == NULL && norm < x->norm)
    append(norm, v, tail);
  else
    before(norm, v, x);
  return 1;
}

void before(ll norm, ll v[DIM], node *place) {
  node *at = (place->prev) ? place->prev : place;
  node *escrow = at->next;
  append(norm, v, &at);
  at->next = escrow;
  escrow->prev = at;
}

void after(ll norm, ll v[DIM], node *place) {
  node *escrow = place->next;
  append(norm, v, &place);
  if ((place->next = escrow))
    escrow->prev = place;
}

void append(ll norm, ll v[DIM], node **tail) {
  (*tail)->next = new_node(norm, v, *tail);
  *tail = (*tail)->next;
  real++;
}

void pop(node **tail) {
  if ((*tail)->prev == NULL)
    return;
  node *tofree = *tail;
  *tail = (*tail)->prev;
  (*tail)->next = NULL;
  free(tofree);
  real--;
}

void squeeze(node **head) {
  if ((*head)->next == NULL)
    return;
  node *tofree = (*head)->next;
  (*head)->next = (*head)->next->next;
  if ((*head)->next)
    (*head)->next->prev = *head;
  free(tofree);
  real--;
}

void free_list(node **tail) {
  while ((*tail)->prev)
    pop(tail);
  free(*tail);
  real--;
  *tail = NULL;
}

void print_list(node *head) {
  ll *x;
  ll norm;
  node *itr = head;
  int dg = -1, cur = 0, j = 0;
  while ((itr = itr->next) != NULL) {
    x = itr->v;
    norm = itr->norm;
    dg = dg < digits(norm) ? digits(norm) : dg;
    for (j = 0; j < DIM; j++)
      if ((cur = digits(x[j])) > dg)
        dg = cur;
  }
  dg += 2;
  while ((head = head->next) != NULL) {
    x = head->v;
    printf("[");
    for (j = 0; j < DIM; j++)
      printf("%*lld", dg, (long long)x[j]);
    printf("] %*lld\n", dg, (long long)ceil(sqrt(head->norm)));
  }
}

node *new_node(ll norm, ll v[DIM], node *prev) {
  node *x = (node *)malloc(sizeof(node));
  if (x != NULL) {
    x->not_ready = 1;
    x->norm = norm;
    memcpy(x->v, v, v ? sizeof(ll) * DIM : 0);
    x->prev = prev;
    x->next = NULL;
  }
  return x;
}

// maybe useless
void cross(node *a, node *b, node **tail) {
  ll t[DIM], tn = 0;
  ll an = a->norm, bn = b->norm;
  double numerator = dot(b->v, a->v, DIM);
  ll m = (ll)round(numerator / bn);
  for (int i = 0; i < DIM; i++) {
    t[i] = a->v[i] - (m * b->v[i]);
    tn += (t[i] * t[i]);
  }
  if (tn < bn)
    place_from(tn, t, b, tail);
  else if (tn < an)
    place_from(tn, t, a, tail);
}

int place(ll norm, ll v[DIM], node **tail) {
  node *x = *tail;
  while (x->prev != NULL && x->norm < norm)
    x = x->prev;
  if (norm == x->norm || norm == 0)
    return 0;
  if (x->next == NULL)
    append(norm, v, tail);
  else
    after(norm, v, x);
  return 1;
}

void replace(ll norm, ll v[DIM], node *from) {
  node *x = from;
  while (x->next != NULL && norm < x->norm)
    x = x->next;
  if (norm == x->norm || norm == 0 || x->prev->prev == NULL)
    return;
  if (x->next == NULL && norm < x->norm)
    set(norm, v, x);
  else
    set(norm, v, x->prev);
}

void set(ll norm, ll v[DIM], node *toset) {
  toset->norm = norm;
  memcpy(toset->v, v, sizeof(ll) * DIM);
}
