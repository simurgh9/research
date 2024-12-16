#include "sieve.h"

int main(int argc, char *argv[]) {
  double beg = period();
  srand(SEED);
  printf("Size: %d\nInflated Size: %d\n", SIZE, INFLATED);

  num B[DIM][DIM];
  norm bnorms[DIM] = {0};
  basis_t(PATH, B, bnorms);
  norm best = initial(B, bnorms, DENSITY);
  double last_mean = mean();
  
  pthread_t threads[THREADS];
  int i = 0, t = 0, unchanged = 0;
  char fmt[] = "%4d %8.3f, %8.3f in %.5fs %s\n";

  do {
    printf(fmt, i++, sqrt(best), sqrt(last_mean), period(), underpop_warn);
    fflush(stdout);

    for (t = 0; t < THREADS; t++) // start sieving
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

  print_vector(P1[order1[0]], DIM, NULL);
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

void basis_t(char path[], num (*B)[DIM], norm bnorms[DIM]) {
  int i = 0, j = 0, c, sign = 1;
  num number = 0;
  FILE *fp = fopen(path, "r");
  if (!fp)
    printf("File %s not found.\n", path);
  while ((c = getc(fp)))
    if (c == '-') {
      sign = -1;
    } else if (c == ',') {
      bnorms[j] += (number * number);
      B[j++][i] = sign * number;
      sign = 1;
      number = 0;
    } else if (c == '\n') {
      bnorms[j] += (number * number);
      B[j][i++] = sign * number;
      j = 0;
      sign = 1;
      number = 0;
    } else if (c == EOF) {
      bnorms[j] += (number * number);
      B[j][i] = sign * number;
      break;
    } else if ('0' >= c || c <= '9') {
      number = (number * 10) + (c - '0');
    }
  fclose(fp);
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
        C[i][j] = (rand() / ((num)RAND_MAX)) < p1;
      } else {
        P1[i][j] = B[i - m][j];
        norms1[i] = bnorms[i - m];
      }
    }
  }

  // P1(m, n) = alpha*[C(m, k)B(k, n)] + beta*P1(m, n)
  num alpha = 1, beta = 0;
  int a = CblasRowMajor, b = CblasNoTrans, c = CblasNoTrans;
  cblas_dgemm(a, b, c, m, n, k, alpha, *C, k, *B, n, beta, *P1, n);
  free(C);

  for (i = 0; i < m; i++)
    norms1[i] = cblas_ddot(n, P1[i], 1, P1[i], 1);

  qsort(order1, SIZE, sizeof(int), compare1);

  for (i = 0; norms1[order1[i]] == 0; i++)
    continue; // find first non-zero norm
  return norms1[order1[i]];
}

void *sieve(void *arg) {
  int id = (int)(uintptr_t)(arg), step = ceil((double)SIZE / THREADS);
  int start = id * step;
  int end = start + step;
  end = end < SIZE ? end : SIZE - 2; // SIZE - 2 due to the look ahead

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

void cross(int i, int j) {
  num t[DIM] = {0};
  norm v1n = norms1[i], v2n = norms1[j];
  num numerator = cblas_ddot(DIM, P1[i], 1, P1[j], 1);
  num m = (num)round(numerator / v1n);
  cblas_dcopy(DIM, P1[j], 1, t, 1);
  cblas_daxpy(DIM, -1 * m, P1[i], 1, t, 1);
  norm tn = cblas_ddot(DIM, t, 1, t, 1);
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

void print_matrix(num (*B)[DIM], int row, int column, int order[]) {
  int dg = -1, cur = 0, i = 0, j = 0;
  for (i = 0; i < row; i++)
    for (j = 0; j < column; j++)
      if ((cur = digits(B[i][j])) > dg)
        dg = cur;
  dg += 2;
  for (i = 0; i < row; i++) {
    for (j = 0; j < column; j++)
      if (order != NULL)
        printf("%*.0f", dg, B[order[i]][j]);
      else
        printf("%*.0f", dg, B[i][j]);
    printf("\n");
  }
}

void print_vector(double *x, int size, int order[]) {
  int dg = -1, cur = 0, i = 0;
  for (i = 0; i < size; i++)
    if ((cur = digits(x[i])) > dg)
      dg = cur;
  dg += 2;
  int len = 70 / dg;
  for (int i = 0; i < size; i += len) {
    for (int j = i; j < i + len && j < size; j++) {
      if (j == 0 && order != NULL)
        printf("[%*.0f", dg - 1, x[order[j]]);
      else if (j == 0 && order == NULL)
        printf("[%*.0f", dg - 1, x[j]);
      else if (j != 0 && order != NULL)
        printf("%*.0f", dg, x[order[j]]);
      else
        printf("%*.0f", dg, x[j]);
    }
    if (i + len < size)
      printf("\n");
    else
      printf("]\n");
  }
}

int digits(num n) {
  if (n == 0)
    return 1;
  return floor(log10(fabs(n))) + 1;
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
  return 0;
}

int compare2(const void *a, const void *b) {
  norm x = norms2[*(int *)a];
  norm y = norms2[*(int *)b];
  if (x > y) return 1;
  if (x < y) return -1;
  return 0;
}

// Local Variables:
// compile-command: "gcc -Wpedantic -Ofast -lblas sieve.c -o sieve.o && \
// ./sieve.o"
// dape-command: (codelldb-cc command-cwd "/Users/tfn/Library/Mobile \
// Documents/com~apple~CloudDocs/grad/research/code/experiment\
// " :program "sieve.o")
// End:
