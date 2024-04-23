// -*- compile-command: "gcc -Wpedantic -Ofast -lblas sieve.c -o sieve.o && ./sieve.o" -*-
#include "sieve.h"

int main(int argc, char *argv[]) {
  double beg = period();
  init();
  ll B[DIM][DIM], norms[DIM] = {0};
  basis(PATH, B, norms);
  initial(B, norms, &tail, 0.01);
  printf("%d things.\n", real);
  sieve();
  printf("Total time: %fs\n", seconds() - beg);
  print_vector(head->next->v, DIM);
  return 0;
}

// helper implements
void init(void) {
  srand(SEED);
  head = new_node(-1, NULL, NULL);
  tail = head;
  for (int i = 0; i < THREADS; i++) {
    H[i] = new_node(-1, NULL, NULL);
    T[i] = H[i];
  }
  H[THREADS] = head, T[THREADS] = tail;
}

void sieve(void) {
  ll best = head->next->norm;
  int i = 0, unchanged = 0;
  char fmt[] = "%4d %8.3f, %8.3f in %.3fs\n";
  do {
    printf(fmt, i++, sqrt(best), mean(head), period());
    fflush(stdout);
    step();
    unchanged++;
    if (best != head->next->norm)
      unchanged = 0;
    best = head->next->norm;
  } while (unchanged < 10);
}

void step(void) {
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

void merge(void) {
  node *h = new_node(-1, NULL, NULL);
  node *t = h;
  ll norm, *v;
  int i = 0, min = -1, k = 0;

  while (k < SIZE) {
    min = -1;
    for (i = 0; i < THREADS + 1; i++) {
      if (H[i]->next == NULL || H[i]->next->norm <= 0)
        continue;
      else if (min < 0)
        min = i;
      else if (H[min]->next->norm > H[i]->next->norm)
        min = i;
    }
    if (min < 0)  // all threads are empty
      break;
    if (H[min]->next->norm != t->norm) {
      after(H[min]->next->norm, H[min]->next->v, t, &t);
      k++;
    }
    squeeze(&H[min], &T[min]);
  }

  if (k < SIZE)
    printf("WARNING: Underpopulation.");
  
  free_list(&T[THREADS]);
  head = h, tail = t;
  H[THREADS] = head, T[THREADS] = tail;
  real = k;
}

void cross(node *v1, node *v2, node **tail) {
  ll t[DIM] = {0}, tn = 0;
  ll v1n = v1->norm, v2n = v2->norm;
  double numerator = dot(v1->v, v2->v, DIM);
  ll m = (ll)round(numerator / v1n);
  cblas_dcopy(DIM, v2->v, 1, t, 1);
  cblas_daxpy(DIM, -1 * m, v1->v, 1, t, 1);
  tn = dot(t, t, DIM);
  // what if just less than tail?
  if (tn != 0 && (tn < v1n || tn < v2n))
    place(tn, t, tail);
}

void initial(ll (*B)[DIM], ll norms[DIM], node **tail, double p1) {
  int i = 0, j = 0, m = 100 * SIZE, k = DIM, n = DIM;
  ll(*P)[DIM] = (ll(*)[DIM])malloc(sizeof(ll[m + DIM][DIM]));
  ll(*C)[DIM] = (ll(*)[DIM])malloc(sizeof(ll[m][DIM]));

  for (i = 0; i < m; i++) {
    for (j = 0; j < DIM; j++) {
      P[i][j] = 0;
      C[i][j] = (rand() / ((ll)RAND_MAX)) < p1;
    }
  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, *C, k, *B,
              n, 1, *P, n);

  free(C);

  for (i = m; i < m + DIM; i++)
    memcpy(P[i], B[i - m], sizeof(B[i - m]));

  int *idx = (int *)malloc((m + DIM) * sizeof(int));
  key = (ll *)malloc((m + DIM) * sizeof(ll));
  for (i = 0; i < m + DIM; i++) {
    idx[i] = i;
    key[i] = i < m ? dot(P[i], P[i], DIM) : norms[i - m];
  }

  qsort(idx, m + DIM, sizeof(int), compare);

  ll last = 0;
  for (i = 0; i < m + DIM; i++) {
    if (key[idx[i]] <= 0)
      continue;
    if (last >= 0 && last == key[idx[i]])
      continue;
    last = key[idx[i]];
    if (real >= SIZE)
      break;
    after(key[idx[i]], P[idx[i]], *tail, tail);
  }
  free(P);
  free(idx);
  free(key);
}

int compare(const void *a, const void *b) {
  return key[*(int *)a] - key[*(int *)b];
}

double dot(ll *x, ll *y, int size) {
  return cblas_ddot(size, x, 1, y, 1);
}

void basis(char path[], ll (*B)[DIM], ll norms[DIM]) {
  int i = 0, j = 0, c, sign = 1;
  ll number = 0;
  FILE *fp = fopen(path, "r");
  if (!fp)
    printf("File %s not found.\n", path);
  while ((c = getc(fp)))
    if (c == '-') {
      sign = -1;
    } else if (c == ',') {
      norms[j] += (number * number);
      B[j++][i] = sign * number;
      sign = 1;
      number = 0;
    } else if (c == '\n') {
      norms[j] += (number * number);
      B[j][i++] = sign * number;
      j = 0;
      sign = 1;
      number = 0;
    } else if (c == EOF) {
      norms[j] += (number * number);
      B[j][i] = sign * number;
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
      printf("%*.0f", dg, B[i][j]);
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
        printf("[%*.0f", dg - 1, x[j]);
      else
        printf("%*.0f", dg, x[j]);
    }
    if (i + len < size)
      printf("\n");
    else
      printf("]\n");
  }
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
    printf("%*.0f [", dg, head->norm);
    for (j = 0; j < DIM; j++)
      printf("%*.0f", dg, x[j]);
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

double seconds(void) {
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return tp.tv_sec + tp.tv_nsec / 1000000000.0;
}

double period(void) {
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
  while (x->prev != NULL && norm < x->norm)
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

void squeeze(node **head, node **tail) {
  if ((*head)->next == NULL)
    return;
  node *tofree = (*head)->next;
  (*head)->next = (*head)->next->next;
  if ((*head)->next != NULL)
    (*head)->next->prev = *head;
  if (tofree == *tail) // list only has two elements
    *tail = *head;
  free(tofree);
}

void free_list(node **tail) {
  while ((*tail)->prev != NULL)
    pop(tail);
  free(*tail);
  *tail = NULL;
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
