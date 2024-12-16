#include "isieve.h"

int main(int argc, char *argv[]) {
  double beg = period();
  srand(SEED);
  printf("Size: %d\nInflated Size: %d\n", SIZE, INFLATED);

  num B[DIM][DIM];
  norm bnorms[DIM] = {0};
  basis_t(B, bnorms);

  norm best = initial(B, bnorms, DENSITY);
  double last_mean = mean();

  // print_vector(norms1, SIZE, order1); // shows overflow
  
  pthread_t threads[THREADS];
  int i = 0, t = 0, unchanged = 0;
  char fmt[] = "%4d %8.3f, %8.3f in %.5fs %s\n";

  do {
    printf(fmt, i++, sqrt(best), sqrt(last_mean), period(), underpop_warn);
    fflush(stdout);

    for (t = 0; t < THREADS; t++)
      pthread_create(&threads[t], NULL, &sieve, (void*)(uintptr_t)t);
    for (t = 0; t < THREADS; t++)
      pthread_join(threads[t], NULL);

    count = 0;
    underpop_warn[0] = '\0';
    selection();
    
    // unchanged = best != norms1[order1[0]] ? 0 : unchanged + 1;
    unchanged = fabs(sqrt(last_mean) - sqrt(mean())) > 1 ? 0 : unchanged + 1;

    best = norms1[order1[0]];
    last_mean = mean();
  } while (unchanged < 10);
  
  print_imag_vector(P1[order1[0]], DIM);
  printf("Total time: %fs\n", seconds() - beg);

  free(P1);
  free(P2);
  free(P3);
  free(norms1);
  free(norms2);
  free(norms3);
  free(order1);
  free(order2);
  return 0;
}

void basis_t(num (*B)[DIM], norm bnorms[DIM]) {
  for (int i = 0; i < DIM; i++) { // START HERE
    bnorms[i] = 0;
    for (int j = 0; j < DIM; j++) {
      double p = 3;  // START HERE: consider lowering because of overflow
      B[i][j] = imaginary(-pow(DIM, p), pow(DIM, p));
      // B[i][j] = imaginary_gaussian(0, pow(DIM, p));
      norm a = (norm) creal(B[i][j]);
      norm b = (norm) cimag(B[i][j]);
      bnorms[i] += ((a * a) + (b * b));
    }
  }
  // B[i] = a*B[(int)uniform(0, DIM - 1)] + B[i]
  num alpha = 1;
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM / 4; j++)
      cblas_zaxpy(DIM, &alpha, B[(int)uniform(0, DIM - 1)], 1, B[i], 1);
    bnorms[i] = pow(cblas_dznrm2(DIM, B[i], 1), 2);
  }
}

norm initial(num (*B)[DIM], norm bnorms[DIM], double p1) {
  int m = SIZE - DIM, k = DIM, n = DIM;
  num(*C)[k] = (num(*)[k])malloc(sizeof(num[m][k]));
  P1 = (num(*)[DIM])malloc(sizeof(num[SIZE][DIM]));
  P2 = (num(*)[DIM])malloc(sizeof(num[INFLATED][DIM]));
  P3 = (num(*)[DIM])malloc(sizeof(num[SIZE][DIM]));
  norms1 = (norm *)malloc(sizeof(num[SIZE]));
  norms2 = (norm *)malloc(sizeof(num[INFLATED]));
  norms3 = (norm *)malloc(sizeof(num[SIZE]));
  order1 = (int *)malloc(sizeof(int[SIZE]));
  order2 = (int *)malloc(sizeof(int[INFLATED]));

  int i = 0, j = 0;
  for (i = 0; i < SIZE; i++) {
    order1[i] = i;
    for (j = 0; j < DIM; j++) {
      if (i < m) {
        norms1[i] = 0;
        double a = (rand() / ((double)RAND_MAX)) < p1;
        double b = (rand() / ((double)RAND_MAX)) < p1;
        C[i][j] = a + b * I;
      } else {
        P1[i][j] = B[i - m][j];
        norms1[i] = bnorms[i - m];
      }
    }
  }

  // P1(m, n) = alpha*[C(m, k)B(k, n)] + beta*P1(m, n)
  num alpha = 1, beta = 0;
  int a = CblasRowMajor, b = CblasNoTrans, c = CblasNoTrans;
  cblas_zgemm(a, b, c, m, n, k, &alpha, *C, k, *B, n, &beta, *P1, n);
  free(C);

  num cur_norm = -1;
  for (i = 0; i < m; i++) { // ATTN: this definitely overflows
    // norms1[i] = pow(cblas_dznrm2(n, P1[i], 1), 2);
    cblas_zdotc_sub(n, P1[i], 1, P1[i], 1, &cur_norm);
    norms1[i] = creal(cur_norm);
  }
  
  // sizeof(int) is correct since order1 is ints
  qsort(order1, SIZE, sizeof(int), compare1);
  
  for (i = 0; norms1[order1[i]] == 0; i++)
    continue; // find first non-zero norm
  return norms1[order1[i]];
}

void *sieve(void *arg) {
  int id = (int)(uintptr_t)(arg), step = ceil((double)SIZE / THREADS);
  int start = id * step;
  int end = start + step; // SIZE - 2 due to the look ahead
  end = end < SIZE ? end : SIZE - 2;

  double ratio = 1;
  for (int i = start; i < end; i++)
    for (int j = i + 1; j < fmin(i + 1 + ratio * SIZE, SIZE); j++)
      if (norms1[order1[i]] == 0 || norms1[order1[i]] == norms1[order1[i + 1]])
        break;
      else if (norms1[order1[j]] == 0)
        continue;
      else if (j < SIZE - 1 && norms1[order1[j]] == norms1[order1[j + 1]])
        continue;
      else if (count >= INFLATED)
        return NULL;
      else
        cross(order1[i], order1[j]); // increments count
  return NULL;
}

void cross(int i, int j) { // start here
  num t[DIM] = {0};
  norm v1n = norms1[i], v2n = norms1[j];
  num numerator = -1;
  cblas_zdotc_sub(DIM, P1[i], 1, P1[j], 1, &numerator);
  num m = -1*(round(creal(numerator) / v1n) + round(cimag(numerator) / v1n)*I);
  cblas_zcopy(DIM, P1[j], 1, t, 1);
  cblas_zaxpy(DIM, &m, P1[i], 1, t, 1);
  num temp_tn = -1;
  cblas_zdotc_sub(DIM, t, 1, t, 1, &temp_tn);
  norm tn  = creal(temp_tn);
  if (tn != 0 && (tn < v1n || tn < v2n)) {
    pthread_mutex_lock(&mutie);
    int old = count++;
    pthread_mutex_unlock(&mutie);
    if (old < INFLATED) {
      norms2[old] = tn;
      memcpy(P2[old], t, sizeof(t));
    }
  }
}

void selection(void) {
  for (int i = 0; i < INFLATED; i++)
    order2[i] = i;
  qsort(order2, INFLATED, sizeof(int), compare2);

  memset(norms3, 0, sizeof(num[SIZE]));
  memset(P3, 0, sizeof(num[SIZE][DIM]));

  int i1 = 0, i2 = 0, i3 = 0;
  while (i3 < SIZE) {
    // printf("%d, %d, %d\n", i1, i2, i3);
    if (i3 < SIZE && i1 >= SIZE && i2 >= INFLATED) {
      sprintf(underpop_warn, "Under population by %5.2f%%!", i3 * 100.0 / SIZE);
      break;
    } else if (i1 < SIZE && norms1[order1[i1]] == 0) {
      i1++;
    } else if (i2 < INFLATED && norms2[order2[i2]] == 0) {
      i2++;
    } else if (i1 < SIZE && i3 > 0 && norms1[order1[i1]] == norms3[i3 - 1]) {
      i1++;
    } else if (i2 < INFLATED && i3 > 0 && norms2[order2[i2]] == norms3[i3 - 1]) {
      i2++;
    } else if (i1 >= SIZE) { // we ran out of norms1
      norms3[i3] = norms2[order2[i2]];
      memcpy(P3[i3], P2[order2[i2]], sizeof(P3[i3]));
      i2++;
      i3++;
    } else if (i2 >= INFLATED) { // we ran out of norms2
      norms3[i3] = norms1[order1[i1]];
      memcpy(P3[i3], P1[order1[i1]], sizeof(P3[i3]));
      i1++;
      i3++;
    } else if (norms1[order1[i1]] <= norms2[order2[i2]]) {
      norms3[i3] = norms1[order1[i1]];
      memcpy(P3[i3], P1[order1[i1]], sizeof(P3[i3]));
      i1++;
      i3++;
    } else if (norms1[order1[i1]] > norms2[order2[i2]]) {
      norms3[i3] = norms2[order2[i2]];
      memcpy(P3[i3], P2[order2[i2]], sizeof(P3[i3]));
      i2++;
      i3++;
    }
  }

  swap((void **)&P1, (void **)&P3);
  swap((void **)&norms1, (void **)&norms3);
  for (int i = 0; i < SIZE; i++)
    order1[i] = i;
}

double mean(void) {
  double k = 0, total = 0;
  for (int i = 0; i < SIZE; i++)
    if (norms1[order1[i]] != 0) {
      k += 1;
      total += norms1[order1[i]];
    }
  return total / k;
}

double uniform(double lo, double hi) {
  return (double)((int)lo + rand() % (int)(hi - lo + 1));
}

num imaginary(double lo, double hi) {
  return uniform(lo, hi) + uniform(lo, hi) * I;
}

num imaginary_gaussian(double mean, double stddev) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;

    // box-muller transform
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    double z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
    z0 = round(z0 * stddev + mean);
    z1 = round(z1 * stddev + mean);

    return z0 + z1 * I;
}

void print_matrix(num (*B)[DIM], int row, int column, int order[]) {
  int i = 0, j = 0, dg = 3;
  for (i = 0; i < row; i++) {
    for (j = 0; j < column; j++)
      if (order != NULL)
        printf("%*.0f%+*.0fi", dg+3, creal(B[order1[i]][j]), dg, cimag(B[order1[i]][j]));
      else
        printf("%*.0f%+*.0fi", dg+3, creal(B[i][j]), dg, cimag(B[i][j]));
    printf("\n");
  }
}

void print_vector(norm *x, int size, int order[]) {
  int dg = -1, cur = 0, i = 0;
  for (i = 0; i < size; i++)
    if ((cur = digits(x[i])) > dg)
      dg = cur;
  dg += 2;
  int len = 70 / dg;
  for (int i = 0; i < size; i += len) {
    for (int j = i; j < i + len && j < size; j++) {
      if (j == 0 && order != NULL)
        printf("[%*llu", dg - 1, x[order[j]]);
      else if (j == 0 && order == NULL)
        printf("[%*llu", dg - 1, x[j]);
      else if (j != 0 && order != NULL)
        printf("%*llu", dg, x[order[j]]);
      else
        printf("%*llu", dg, x[j]);
    }
    if (i + len < size)
      printf("\n");
    else
      printf("]\n");
  }
}

void print_imag_vector(num *x, int size) {
  int dg = 3;
  for (int i = 0; i < size; i++)
    printf("%*.0f%+*.0fi\t", dg, creal(x[i]), dg, cimag(x[i]));
  printf("\n");
}

int digits(norm n) {
  if (n == 0)
    return 1;
  return floor(log10((n))) + 1;
}

void swap(void **a, void **b) {
  void *escrow = *a;
  *a = *b;
  *b = escrow;
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

double seconds(void) {
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return tp.tv_sec + tp.tv_nsec / 1000000000.0;
}

int compare1(const void *a, const void *b) {
  norm x = norms1[*(int *)a];
  norm y = norms1[*(int *)b];
  if (x > y) return 1;
  if (x < y) return -1;
  return 0; // return x - y;
}

int compare2(const void *a, const void *b) {
  norm x = norms2[*(int *)a];
  norm y = norms2[*(int *)b];
  if (x > y) return 1;
  if (x < y) return -1;
  return 0; // return x - y;
}

// Local Variables:
// compile-command: "gcc -Wpedantic -Ofast -lblas isieve.c -o isieve.o && \
// ./isieve.o"
// End:
