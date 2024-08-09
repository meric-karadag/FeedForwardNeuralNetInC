#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"

int main()
{
  // Seed the random number generator
  srand((unsigned int)time(NULL));

  // Test matrix creation and filling
  Matrix *matrix = matrixCreate(4, 3);
  matrixFill(matrix, 5.0);
  printf("Matrix after filling with 5.0:\n");
  matrixPrint(matrix);
  printf("\n");

  // Test matrix copying
  Matrix *matrixCopyTest = matrixCopy(matrix);
  printf("Copy of the filled matrix:\n");
  matrixPrint(matrixCopyTest);
  printf("\n");

  // Test random initialization
  matrixRandomInit(matrix, 0.0, 0.5);
  printf("Matrix after random initialization (mean = 0, stddev = 1):\n");
  matrixPrint(matrix);
  printf("\n");

  // Test saving to file
  matrixSave(matrix, "matrix.txt");
  printf("Matrix saved to 'matrix.txt'.\n");

  // Test loading from file
  Matrix *loadedMatrix = matrixLoad("matrix.txt");
  printf("Matrix loaded from 'matrix.txt':\n");
  matrixPrint(loadedMatrix);
  printf("\n");

  // Test matrix argmax
  Matrix *columnVector = matrixCreate(3, 1);
  columnVector->entries[0][0] = 0.1;
  columnVector->entries[1][0] = 0.5;
  columnVector->entries[2][0] = 0.3;
  int maxIdx = matrixArgmax(columnVector);
  printf("Argmax of column vector (should be 1): %d\n", maxIdx);
  printf("\n");

  // Test matrix flatten
  Matrix *flattened0 = matrixFlatten(matrix, 0);
  printf("Matrix flattened (axis = 0):\n");
  matrixPrint(flattened0);
  printf("\n");

  Matrix *flattened1 = matrixFlatten(matrix, 1);
  printf("Matrix flattened (axis = 1):\n");
  matrixPrint(flattened1);
  printf("\n");

  // Free all matrices
  matrixFree(matrix);
  matrixFree(matrixCopyTest);
  matrixFree(loadedMatrix);
  matrixFree(columnVector);
  matrixFree(flattened0);
  matrixFree(flattened1);

  return 0;
}