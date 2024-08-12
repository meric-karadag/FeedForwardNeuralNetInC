#ifndef MATRIXOPS_H
#define MATRIXOPS_H

#include "matrix.h"

Matrix *elementWiseMultiply(Matrix *matrix1, Matrix *matrix2);
Matrix *elementWiseAdd(Matrix *matrix1, Matrix *matrix2);
Matrix *elementWiseSubtract(Matrix *matrix1, Matrix *matrix2);
Matrix *dot(Matrix *matrix1, Matrix *matrix2);
Matrix *transpose(Matrix *matrix);
Matrix *apply(Matrix *matrix, double (*func)(double));
Matrix *scale(Matrix *matrix, double scalar);
Matrix *addScalar(Matrix *matrix, double scalar);

#endif