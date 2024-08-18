#include "activations.h"
#include <math.h>
#include "../matrix/matrixOps.h"

double sigmoid(double x)
{
  return 1.0 / (1.0 + exp(-1.0 * x));
}

double relu(double x)
{
  if (x >= 0)
  {
    return x;
  }
  return 0.0;
}

Matrix *sigmoidPrime(Matrix *matrix)
{
  Matrix *ones = matrixCreate(matrix->rows, matrix->cols);
  matrixFill(ones, 1.0);

  Matrix *subtracted = elementWiseSubtract(ones, matrix);
  Matrix *multiplied = elementWiseMultiply(matrix, subtracted);

  matrixFree(ones);
  matrixFree(subtracted);

  return multiplied;
}

Matrix *reluPrime(Matrix *matrix)
{
  int i, j;
  Matrix *result = matrixCreate(matrix->rows, matrix->cols);

  for (i = 0; i < matrix->rows; i++)
  {
    for (j = 0; j < matrix->cols; j++)
    {
      if (matrix->entries[i][j] >= 0)
      {
        result->entries[i][j] = 1;
      }
      else
      {
        result->entries[i][j] = 0;
      }
    }
  }
  return result;
}

Matrix *softmax(Matrix *matrix)
{
  int i, j;
  double total = 0;
  Matrix *result = matrixCreate(matrix->rows, matrix->cols);

  for (i = 0; i < matrix->rows; i++)
  {
    for (j = 0; j < matrix->cols; j++)
    {
      total += exp(matrix->entries[i][j]);
    }
  }

  for (i = 0; i < matrix->rows; i++)
  {
    for (j = 0; j < matrix->cols; j++)
    {
      matrix->entries[i][j] = exp(matrix->entries[i][j]) / total;
    }
  }

  return result;
}