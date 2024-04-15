// -*- compile-command: "gcc -Wpedantic -Ofast -lblas sieve.c -o sieve.o && ./sieve.o" -*-
#include "sieve.h"

int main(int argc, char *argv[]) {
  double beg = period();
  init();
  ll B[DIM][DIM];
  basis(PATH, B);
  initial(B, &tail, 0.01);
  sieve();
  printf("Total time: %fs\n", seconds() - beg);
  print_vector(tail->v, DIM);
  return 0;
}

// helper implements
void init() {
  srand(SEED);
  head = new_node(-1, NULL, NULL);
  tail = head;
  for (int i = 0; i < THREADS; i++) {
    H[i] = new_node(-1, NULL, NULL);
    T[i] = H[i];
  }
}

void sieve() {
  ll best = tail->norm;
  int i = 0, unchanged = 0;
  char fmt[] = "%4d %8.3f, %8.3f in %.3fs\n";
  do {
    printf(fmt, i++, sqrt(best), mean(head), period());
    fflush(stdout);
    step();
    unchanged++;
    if (best != tail->norm)
      unchanged = 0;
    best = tail->norm;
  } while (unchanged < 10);
}

void step() {
  int indices[THREADS], i = 0;
  pthread_t threads[THREADS];
  for (i = 0; i < THREADS; i++)
    indices[i] = i;
  for (i = 0; i < THREADS; i++)
    pthread_create(&threads[i], NULL, &job, &indices[i]);
  for (i = 0; i < THREADS; i++)
    pthread_join(threads[i], NULL);
  merge();
}

void *job(void *arg) {
  int j, i = *(int *)(arg), step = SIZE / THREADS;
  int start = i * step;
  int end = start + step;

  if (step <= 0 && i > 0)
    return NULL;
  else if (i == THREADS - 1 || step <= 0 && i == 0)
    end = SIZE;
  
  while (T[i]->prev != NULL) // clean from last time
    pop(&T[i]);
  
  node *a = head, *b;
  for (j = 0; j < start; j++)
    a = a->next;
  while ((a = a->next) != NULL && j++ < end && real < 5 * SIZE) {
    b = a;
    while ((b = b->next) != NULL && real < 5 * SIZE)
      cross(a, b, &T[i]);
  }
  // printf("Thread %d (%d:%d) is done.\n", i, start, end);
  return NULL;
}

void merge() {
  node *h = new_node(-1, NULL, NULL);
  node *t = h;
  ll norm, *v;
  int i = 0, k = 0, argmin = -1;
  while (k++ < SIZE) {
    argmin = -1, norm = tail->norm, v = tail->v;
    for (i = 0; i < THREADS; i++)
      if (T[i]->norm > 0 && norm > T[i]->norm)
        argmin = i, norm = T[i]->norm, v = T[i]->v;
    after(norm, v, h, &t);
    if (argmin > -1)
      pop(&T[argmin]);
    else
      pop(&tail);
  }

  free_list(&tail);
  head = h, tail = t;
  real = SIZE;
}

void cross(node *a, node *b, node **tail) {
  ll t[DIM], tn = 0;
  ll an = a->norm, bn = b->norm;
  double numerator = dot(b->v, a->v, DIM);
  ll m = (ll)round(numerator / bn);
  cblas_dcopy(DIM, a->v, 1, t, 1);
  cblas_daxpy(DIM, -1 * m, b->v, 1, t, 1);
  tn = dot(t, t, DIM);
  if (tn < bn)
    place(tn, t, tail);
  else if (tn < an)
    place(tn, t, tail);
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
}

void random_bits(ll *fill, int size, double p1) {
  for (int i = 0; i < size; i++)
    fill[i] = (rand() / ((double)RAND_MAX)) < p1;
}

double dot(ll *x, ll *y, int size) { return cblas_ddot(size, x, 1, y, 1); }

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
    return timestamp;
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

int delta_real(int c) {
  pthread_mutex_lock(&mutie);
  real += c;
  pthread_mutex_unlock(&mutie);
  return real;
}


// list implements
int place(ll norm, ll v[DIM], node **tail) {
  node *x = *tail;
  while (x->prev != NULL && x->norm < norm)
    x = x->prev;
  if (norm == x->norm || norm == 0)
    return 0;
  after(norm, v, x, tail);
  return 1;
}

int place_from(ll norm, ll v[DIM], node *from, node **tail) {
  node *x = from;
  while (x->next != NULL && norm < x->norm)
    x = x->next;
  if (norm == x->norm || norm == 0)
    return 0;
  if (x->next == NULL && norm < x->norm)
    after(norm, v, x, tail);
  else
    before(norm, v, x, tail);
  return 1;
}

void before(ll norm, ll v[DIM], node *mid, node **tail) {
  node *at = (mid->prev) != NULL ? mid->prev : mid;
  after(norm, v, at, tail);
}

void after(ll norm, ll v[DIM], node *mid, node **tail) {
  node **at = (mid == *tail) ? tail : &mid;
  node *escrow = (*at)->next;
  (*at)->next = new_node(norm, v, *at);
  *at = (*at)->next;
  (*at)->next = escrow;
  if (escrow != NULL)
    escrow->prev = *at;
  delta_real(1);
}

void pop(node **tail) {
  if ((*tail)->prev == NULL)
    return;
  node *tofree = *tail;
  *tail = (*tail)->prev;
  (*tail)->next = NULL;
  free(tofree);
}

void squeeze(node **head) {
  if ((*head)->next == NULL)
    return;
  node *tofree = (*head)->next;
  (*head)->next = (*head)->next->next;
  if ((*head)->next)
    (*head)->next->prev = *head;
  free(tofree);
}

void free_list(node **tail) {
  while ((*tail)->prev != NULL)
    pop(tail);
  free(*tail);
  *tail = NULL;
}

void print_norms(node *head) {
  if (head->next == NULL)
    printf("\n");
  while ((head = head->next) != NULL)
    if (head->next != NULL)
      printf("%.0f, ", head->norm);
    else
      printf("%.0f\n", head->norm);
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
    printf("%*lld [", dg, (long long)head->norm);
    for (j = 0; j < DIM; j++)
      printf("%*lld", dg, (long long)x[j]);
    printf("]\n");
  }
}

node *new_node(ll norm, ll v[DIM], node *prev) {
  node *x = (node *)malloc(sizeof(node));
  if (x != NULL) {
    x->norm = norm;
    memcpy(x->v, v, v ? sizeof(ll) * DIM : 0);
    x->prev = prev;
    x->next = NULL;
  }
  return x;
}

// maybe useless
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
