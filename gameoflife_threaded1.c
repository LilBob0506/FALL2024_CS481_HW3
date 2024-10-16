/* 
   Solution to the game of life program.
   Author: Purushotham Bangalore
   Date: Feb 17, 2010

   Use -DDEBUG1 for output at the start and end.
   Use -DDEBUG2 for output at each iteration.
*/

/*
Name: Shan Sahib
Email: sssahib@crimson.ua.edu
Course Section: CS 481-001
Homework #: 3
Instructions to compile the program: gcc -O gameoflife_threaded1.c -o gameoflife (to print add -DDEBUG_PRINT1 -DDEBUG_PRINT2 -DDEBUG_PRINT0 accordingly)
Instructions to run the program: ./gameoflife <size> <max number of generations> <number of threads> <output file directory>  
 */

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>
#include <math.h>

#define DIES   0
#define ALIVE  1

double gettime(void) {
  struct timeval tval;

  gettimeofday(&tval, NULL);

  return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}

int **allocarray(int P, int Q) {
  int i, *p, **a;

  p = (int *)malloc(P*Q*sizeof(int));
  a = (int **)malloc(P*sizeof(int*));
  for (i = 0; i < P; i++)
    a[i] = &p[i*Q]; 

  return a;
}

void freearray(int **a) {
  free(&a[0][0]);
  free(a);
}

/*
void printarray(int **a, int N, int k) {
  int i, j;
  printf("Life after %d iterations:\n", k) ;
  for (i = 1; i < N+1; i++) {
    for (j = 1; j< N+1; j++)
      printf("%d ", a[i][j]);
    printf("\n");
  }
  printf("\n");
}
*/

void printarray(char* outputDirectory, int **a, int N, int k) {
  int i, j;

  FILE *file = fopen(outputDirectory, "w");

  if (file == NULL) {
    printf("Failed to create the file.\n");
    return;
  }

  printf("Life after %d iterations:\n", k) ;
  for (i = 1; i < N+1; i++) {
    for (j = 1; j< N+1; j++)
      fprintf(file, "%d ", a[i][j]);
    fprintf(file, "\n");
  }
  fprintf(file, "\n");
}

int compute(int **life, int **temp, int N, int P, int Q, int NTIMES) {
  int i, j, k, value;
  int flag = 0; 

  #pragma omp parallel default(none) shared(life, temp, N, P, Q, NTIMES, flag) private(i, j, k, value) num_threads(P * Q) 
  {
    for (k = 0; k < NTIMES; k++) {
      int local_flag = 0;  

      int tid = omp_get_thread_num();
      int p = tid / Q;
      int q = tid % Q;

      
      int myM = N / P;
      int istart = p * myM;
      int iend = istart + myM;
      if (p == P - 1) iend = N;  

      #ifdef DEBUG0
        printf("tid=%d istart=%d iend=%d\n",tid,istart,iend);
      #endif

      for (i = istart + 1; i < iend + 1; i++) {
        int myN = N / Q;
        int jstart = q * myN;
        int jend = jstart + myN;
        if (q == Q - 1) jend = N;  
    
        #ifdef DEBUG0
            printf("tid=%d[p,q]=[%d,%d]: {istart,iend}:{%d,%d} {jstart,jend}:{%d,%d}\n", tid, p, q, istart, iend, jstart, jend);
        #endif
        for (j = jstart + 1; j < jend + 1; j++) {
          /* find out the value of the current cell */
          value = life[i - 1][j - 1] + life[i - 1][j] + life[i - 1][j + 1]
          + life[i][j - 1] + life[i][j + 1]
          + life[i + 1][j - 1] + life[i + 1][j] + life[i + 1][j + 1];

          /* check if the cell dies or life is born */
          if (life[i][j]) {  // cell was alive in the earlier iteration
            if (value < 2 || value > 3) {
              temp[i][j] = DIES;
              local_flag++;  // value changed 
            } 
          else 
            temp[i][j] = ALIVE;  // No change      
          } 
          else {  // cell was dead in the earlier iteration
            if (value == 3) {
              temp[i][j] = ALIVE;
              local_flag++;  // Mark that a change occurred
            } 
          else 
            temp[i][j] = DIES;  // No change
          }
        }
      }

      #pragma omp critical
      {
        flag += local_flag; 
      }

      #pragma omp barrier

      // Stop if no changes were made
      #pragma omp single
      {
        if (flag == 0) {
          NTIMES=k; 
        } 
        else {
          flag = 0; 
        }
      }

      // Copy the new values to the old array
      #pragma omp for
      for (int ii = 1; ii < N + 1; ii++) {
        for (int jj = 1; jj < N + 1; jj++) {
          life[ii][jj] = temp[ii][jj];
        }
      }

      #ifdef DEBUG2
      // Print debug information using a single thread
      #pragma omp single
      {
        printf("No. of cells whose value changed in iteration %d = %d\n", k + 1, flag);
        // printarray(life, N, k + 1);
        printarray(outputDirectory,life, N, k+1);
      }
      #endif
    }
  }

  return k;
}


int main(int argc, char **argv) {
  int N, NTIMES, **life=NULL, **temp=NULL, **ptr;
  int i, j, k, flag = 0, value;
  double t1, t2;
  int P, Q, threadNum;
  char *outputDirectory;

  if (argc != 5) {
    printf("Usage: %s <size> <max. iterations> <number of threads> <output file directory>\n", argv[0]);
    exit(-1);
  }

  N = atoi(argv[1]);
  NTIMES = atoi(argv[2]);
  threadNum = atoi(argv[3]);
  outputDirectory = argv[4];

  P = 1;
  Q = threadNum;

  /* Allocate memory for both arrays */
  life = allocarray(N+2,N+2);
  temp = allocarray(N+2,N+2);

  /* Initialize the boundaries of the life matrix */
  for (i = 0; i < N+2; i++) {
    life[0][i] = life[i][0] = life[N+1][i] = life[i][N+1] = DIES ;
    temp[0][i] = temp[i][0] = temp[N+1][i] = temp[i][N+1] = DIES ;
  }

  /* Initialize the life matrix */
  for (i = 1; i < N+1; i++) {
    srand(54321|i);
    for (j = 1; j< N+1; j++)
      if (drand48() < 0.5) 
	      life[i][j] = ALIVE ;
      else
	      life[i][j] = DIES ;
  }

  #ifdef DEBUG1
    /* Display the initialized life matrix */
    // printarray(life, N, 0);
    printarray(outputDirectory,life, N, 0);
  #endif

  t1 = gettime();
  k = compute(life, temp, N, P, Q, NTIMES);
  t2 = gettime();

  printf("Time taken %f seconds for %d iterations\n", t2 - t1, NTIMES);

  #ifdef DEBUG1
    /* Display the life matrix after k iterations */
    // printarray(life, N, k);
    printarray(outputDirectory,life, N, k);
  #endif

  freearray(life);
  freearray(temp);

  printf("Program terminates normally\n") ;

  return 0;
}
