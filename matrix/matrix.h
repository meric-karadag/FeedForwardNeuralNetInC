#pragma once

typedef struct
{
  double **entries;
  int rows;
  int cols;
} Matrix;

Matrix *matrixCreate(int row, int col);
void matrixFill(Matrix *matrix, double num);
void matrixFree(Matrix *matrix);
void matrixPrint(Matrix *matrix);
Matrix *matrixCopy(Matrix *matrix);
void matrixSave(Matrix *matrix, char *fileName);
Matrix *matrixLoad(char *fileName);
double randn(double mean, double std);
void matrixRandomInit(Matrix *matrix, double mean, double std);
int matrixArgmax(Matrix *matrix);
Matrix *matrixFlatten(Matrix *matrix, int axis);