#include "img.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXCHAR 10000

Img **csvToImg(char *filename, int numImages, int imgDimRow, int imgDimCol)
{
  FILE *file;
  int i = 0, j = 0;

  Img **imgs = malloc(numImages * sizeof(Img *));

  char row[MAXCHAR];
  file = fopen(filename, "r");

  // Read the unnecessary first file of .csv file
  fgets(row, MAXCHAR, file);

  // Read until end of file or reaching numImages images
  while (!feof(file) && i < numImages)
  {
    // Allocate necessary memory to store image
    imgs[i] = malloc(sizeof(Img));
    imgs[i]->imgData = matrixCreate(imgDimRow, imgDimCol);

    j = 0;

    // Retrieve the first token in the row which stands for the label
    fgets(row, MAXCHAR, file);
    char *token = strtok(row, ",");
    imgs[i]->label = atoi(token);

    token = strtok(NULL, ",");
    while (token)
    {
      imgs[i]->imgData->entries[j / imgDimCol][j % imgDimCol] = (double)atoi(token) / 256.0;
      token = strtok(NULL, ",");
      j++;
    }
    i++;
  }
  fclose(file);
  return imgs;
}

void imgPrint(Img *img)
{
  printf("Img label: %d\n", img->label);
  matrixPrint(img->imgData);
}

void imgFree(Img *img)
{
  matrixFree(img->imgData);
  free(img);
  img = NULL;
}

void imgsFree(Img **imgs, int n)
{
  for (int i = 0; i < n; i++)
  {
    imgFree(imgs[i]);
  }
  free(imgs);
  imgs = NULL;
}