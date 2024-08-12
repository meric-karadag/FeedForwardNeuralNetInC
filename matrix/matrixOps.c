#include <stdio.h>
#include <stdlib.h>
#include "matrixOps.h"

int checkDimensions(Matrix *matrix1, Matrix *matrix2)
{
  return (matrix1->rows == matrix2->rows && matrix1->cols == matrix2->cols);
}

Matrix *elementWiseMultiply(Matrix *matrix1, Matrix *matrix2)
{
  if (!checkDimensions(matrix1, matrix2))
  {
    printf("Matrix dimensions do not match for multiplication: (%d, %d), (%d, %d)\n",
           matrix1->rows, matrix1->cols, matrix2->rows, matrix2->cols);
    exit(EXIT_FAILURE);
  }

  Matrix *result = matrixCreate(matrix1->rows, matrix1->cols);
  for (int i = 0; i < matrix1->rows; i++)
  {
    for (int j = 0; j < matrix1->cols; j++)
    {
      result->entries[i][j] = matrix1->entries[i][j] * matrix2->entries[i][j];
    }
  }
  return result;
}

Matrix *elementWiseAdd(Matrix *matrix1, Matrix *matrix2)
{
  if (!checkDimensions(matrix1, matrix2))
  {
    printf("Matrix dimensions do not match for addition: (%d, %d), (%d, %d)\n",
           matrix1->rows, matrix1->cols, matrix2->rows, matrix2->cols);
    exit(EXIT_FAILURE);
  }

  Matrix *result = matrixCreate(matrix1->rows, matrix1->cols);
  for (int i = 0; i < matrix1->rows; i++)
  {
    for (int j = 0; j < matrix1->cols; j++)
    {
      result->entries[i][j] = matrix1->entries[i][j] + matrix2->entries[i][j];
    }
  }
  return result;
}

Matrix *elementWiseSubtract(Matrix *matrix1, Matrix *matrix2)
{
  if (!checkDimensions(matrix1, matrix2))
  {
    printf("Matrix dimensions do not match for subtraction(%d, %d), (%d, %d)\n",
           matrix1->rows, matrix1->cols, matrix2->rows, matrix2->cols);
    exit(EXIT_FAILURE);
  }

  Matrix *result = matrixCreate(matrix1->rows, matrix1->cols);
  for (int i = 0; i < matrix1->rows; i++)
  {
    for (int j = 0; j < matrix1->cols; j++)
    {
      result->entries[i][j] = matrix1->entries[i][j] - matrix2->entries[i][j];
    }
  }
  return result;
}

Matrix *apply(Matrix *matrix, double (*func)(double))
{
  Matrix *result = matrixCreate(matrix->rows, matrix->cols);
  for (int i = 0; i < matrix->rows; i++)
  {
    for (int j = 0; j < matrix->cols; j++)
    {
      result->entries[i][j] = func(matrix->entries[i][j]);
    }
  }
  return result;
}

Matrix *dot(Matrix *matrix1, Matrix *matrix2)
{
  if (matrix1->cols != matrix2->rows)
  {
    printf("Matrix dimensions do not match for matrix multiplication(%d, %d), (%d, %d)\n",
           matrix1->rows, matrix1->cols, matrix2->rows, matrix2->cols);
    exit(EXIT_FAILURE);
  }

  int i, j, k;
  double sum;
  Matrix *result = matrixCreate(matrix1->rows, matrix2->cols);

  for (i = 0; i < matrix1->rows; i++)
  {
    for (j = 0; j < matrix2->cols; j++)
    {
      sum = 0.0;
      for (k = 0; k < matrix1->cols; k++)
      {
        sum += matrix1->entries[i][k] * matrix2->entries[k][j];
      }
      result->entries[i][j] = sum;
    }
  }
  return result;
}

Matrix *scale(Matrix *matrix, double scalar)
{
  int i, j;
  Matrix *result = matrixCopy(matrix);

  for (i = 0; i < matrix->rows; i++)
  {
    for (j = 0; j < matrix->cols; j++)
    {
      result->entries[i][j] *= scalar;
    }
  }
  return result;
}

Matrix *addScalar(Matrix *matrix, double scalar)
{
  int i, j;
  Matrix *result = matrixCopy(matrix);

  for (i = 0; i < matrix->rows; i++)
  {
    for (j = 0; j < matrix->cols; j++)
    {
      result->entries[i][j] += scalar;
    }
  }
  return result;
}

Matrix *transpose(Matrix *matrix)
{
  int i, j;
  Matrix *result = matrixCreate(matrix->cols, matrix->rows);

  for (i = 0; i < matrix->rows; i++)
  {
    for (j = 0; j < matrix->cols; j++)
    {
      result->entries[j][i] = matrix->entries[i][j];
    }
  }
  return result;
}