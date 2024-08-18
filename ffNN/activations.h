#ifndef ACTIVATIONS
#define ACTIVATIONS
#include "../matrix/matrix.h"
#include "../matrix/matrixOps.h"

double sigmoid(double x);
double relu(double x);

Matrix *softmax(Matrix *matrix);
Matrix *sigmoidPrime(Matrix *matrix);
Matrix *reluPrime(Matrix *matrix);
#endif