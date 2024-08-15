#ifndef IMAGE
#define IMAGE
#include "../matrix/matrix.h"

typedef struct img
{
  Matrix *imgData;
  int label;
} Img;

Img **csvToImg(char *filename, int *numImages);
void imgPrint(Img *img);
void imgFree(Img *img);
void imgsFree(Img **imgs, int numImages);

#endif