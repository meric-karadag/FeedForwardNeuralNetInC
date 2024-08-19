#include "activations.h"
#include <math.h>
#include "../matrix/matrixOps.h"

double square(double x)
{
  return x * x;
}

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

double reluPrime(double x)
{
  return x >= 0 ? 1 : 0;
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

// Apply softmax row-wise, normalize each row on its own
Matrix *softmax(Matrix *matrix)
{
  int i, j;
  double sumExp;
  Matrix *result = matrixCreate(matrix->rows, matrix->cols);

  for (i = 0; i < matrix->rows; i++)
  {
    sumExp = 0;
    for (j = 0; j < matrix->cols; j++)
    {
      sumExp += exp(matrix->entries[i][j]);
    }
    for (j = 0; j < matrix->cols; j++)
    {
      result->entries[i][j] = exp(matrix->entries[i][j]) / sumExp;
    }
  }
  return result;
}

double mseLoss(Matrix *preds, Matrix *groundTruth)
{
  double mse, sum = 0.0;
  int i, j;
  Matrix *diff = elementWiseSubtract(preds, groundTruth);
  Matrix *squaredDiff = apply(diff, square);

  for (i = 0; i < squaredDiff->rows; i++)
  {
    for (j = 0; j < squaredDiff->cols; j++)
    {
      sum += squaredDiff->entries[i][j];
    }
  }
  mse = sum / squaredDiff->rows;
  matrixFree(diff);
  matrixFree(squaredDiff);
  return mse;
}

double crossEntropyLoss(Matrix *preds, Matrix *groundTruth)
{
  double loss = 0;
  int i, j;
  for (i = 0; i < preds->rows; i++)
  {
    for (j = 0; j < preds->cols; j++)
    {
      loss -= groundTruth->entries[i][j] * log(preds->entries[i][j]);
    }
  }
  return loss / preds->rows;
