#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXCHAR 512
// Allocate necessary space for row x col matrix
Matrix *matrixCreate(int row, int col)
{
  Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
  matrix->rows = row;
  matrix->cols = col;

  // Allocate memory for double pointers for each row
  matrix->entries = malloc(row * sizeof(double *));

  // Allocate col number of doubles for each row
  for (int i = 0; i < row; i++)
  {
    matrix->entries[i] = malloc(sizeof(double) * row);
  }

  return matrix;
}

// Fill each element of matrix by num
void matrixFill(Matrix *matrix, double num)
{
  for (int i = 0; i < matrix->rows; i++)
  {
    for (int j = 0; j < matrix->cols; j++)
    {
      matrix->entries[i][j] = num;
    }
  }
}

// Free space consumed by matrix
void matrixFree(Matrix *matrix)
{
  for (int i = 0; i < matrix->rows; i++)
  {
    free(matrix->entries[i]);
  }
  free(matrix->entries);
  free(matrix);
  matrix = NULL;
}

// Print shape and entries of matrix like a numpy array
void matrixPrint(Matrix *matrix)
{
  printf("Matrix shape: (%d,%d)\n", matrix->rows, matrix->cols);
  printf("Matrix entries:\n[");
  for (int i = 0; i < matrix->rows; i++)
  {
    printf("[");
    for (int j = 0; j < matrix->cols - 1; j++)
    {
      printf(" %.3f,", matrix->entries[i][j]);
    }
    if (i != matrix->rows - 1)
      printf(" %.3f]\n", matrix->entries[i][matrix->cols - 1]);
    else
      printf(" %.3f]", matrix->entries[i][matrix->cols - 1]);
  }
  printf("]");
}

// Allocate memory and create a deepcopy of matrix
Matrix *matrixCopy(Matrix *matrix)
{
  int i, j;
  Matrix *copy = matrixCreate(matrix->rows, matrix->cols);
  for (i = 0; i < matrix->rows; i++)
  {
    for (j = 0; j < matrix->cols; j++)
    {
      copy->entries[i][j] = matrix->entries[i][j];
    }
  }
  return copy;
}

// Save contents of matrix to file (Correct up to 8 decimal points)
void matrixSave(Matrix *matrix, char *fileName)
{
  int i, j;
  FILE *file = fopen(fileName, "w");

  // First write num rows and cols to make it easier to read from file
  fprintf(file, "%d\n", matrix->rows);
  fprintf(file, "%d\n", matrix->cols);

  // Write entries one by one sequentially
  for (i = 0; i < matrix->rows; i++)
  {
    for (j = 0; j < matrix->cols; j++)
    {
      fprintf(file, "%.8f\n", matrix->entries[i][j]);
    }
  }
  printf("Saved contents of matrix to %s\n", fileName);
  fclose(file);
}

// Loads matrix contents from a saved file
Matrix *matrixLoad(char *fileName)
{
  int i, j;
  int rows, cols;
  FILE *file = fopen(fileName, "r");
  char entry[MAXCHAR];

  // Read first line indicating num rows
  fgets(entry, MAXCHAR, file);
  rows = atoi(entry);

  // Read second line indicating num cols
  fgets(entry, MAXCHAR, file);
  cols = atoi(entry);

  // Initialize matrix with correct shape
  Matrix *matrix = matrixCreate(rows, cols);

  // Read the entries one by one and copy to matrix's entries
  for (i = 0; i < rows; i++)
  {
    for (j = 0; j < cols; j++)
    {
      fgets(entry, MAXCHAR, file);
      matrix->entries[i][j] = strtod(entry, NULL);
    }
  }

  printf("Matrix is succesfully loaded from file '%s'\n", fileName);
  fclose(file);

  return matrix;
}

// Generate random numbers from normal distribution Using Box-Muller Transofrm
// Pass 0 mean, 1 std for standard uniform variables
double randn(double mean, double std)
{
  double u1 = ((double)rand() / RAND_MAX);
  double u2 = ((double)rand() / RAND_MAX);

  double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
  // double z0 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

  return mean + std * z0;
}

// Randomly initializes entries of matrix sampled from normal distribution with given args
// Pass 0 mean, 1 std for standard uniform variables
void matrixRandomInit(Matrix *matrix, double mean, double std)
{
  int i, j;
  for (i = 0; i < matrix->rows; i++)
  {
    for (j = 0; j < matrix->cols; j++)
    {
      matrix->entries[i][j] = randn(mean, std);
    }
  }
}

// Returns the index of maximum value from a column vector: Matrix of size (N x 1)
int matrixArgmax(Matrix *matrix)
{
  int i, maxIdx = 0;
  double maxVal = __DBL_MIN__;

  for (i = 0; i < matrix->rows; i++)
  {
    if (matrix->entries[i][0] > maxVal)
    {
      maxVal = matrix->entries[i][0];
      maxIdx = i;
    }
  }

  return maxIdx;
}

// Takes a (N,M) matrix as input, returns flattened version of it
// If Axis = 0 -> Returns (N*M, 1) Column vector
// If Axis = 1 -> Returns(1, N*M) Column vector
Matrix *matrixFlatten(Matrix *matrix, int axis)
{
  int i, j;
  Matrix *flattened;
  if (axis == 0)
  {
    flattened = matrixCreate(matrix->rows * matrix->cols, 1);
    for (i = 0; i < matrix->rows; i++)
    {
      for (j = 0; j < matrix->cols; j++)
      {
        flattened->entries[i * matrix->cols + j][0] = matrix->entries[i][j];
      }
    }
    return flattened;
  }
  else if (axis == 1)
  {
    flattened = matrixCreate(1, matrix->rows * matrix->cols);
    for (i = 0; i < matrix->rows; i++)
    {
      for (j = 0; j < matrix->cols; j++)
      {
        flattened->entries[0][i * matrix->cols + j] = matrix->entries[i][j];
      }
    }
    return flattened;
  }
  else
  {
    printf("Argument to matrix_flatten must be 0 or 1");
    exit(EXIT_FAILURE);
  }
}