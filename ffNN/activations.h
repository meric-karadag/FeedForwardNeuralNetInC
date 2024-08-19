#ifndef ACTIVATIONS
#define ACTIVATIONS
#include "../matrix/matrix.h"
#include "../matrix/matrixOps.h"

double square(double x);
double sigmoid(double x);
double relu(double x);
double reluPrime(double x);

Matrix *softmax(Matrix *matrix);
Matrix *sigmoidPrime(Matrix *matrix);
double mseLoss(Matrix *preds, Matrix *groundTruth);
double crossEntropyLoss(Matrix *preds, Matrix *groundTruth);
#endif