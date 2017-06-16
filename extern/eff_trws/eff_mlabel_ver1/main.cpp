
/***  Please see README.txt  ***/

#include <cstdio>
#include "image.h"

int main()
{
  image* Image;

  int num_labels = 4; // number of labels

  // NOTE: Write code to initialize the energy parameters in image::image(char*, int) 
  Image = new image("file.ppm", num_labels);

  Image -> kovtun(true);
  //Image -> kovtun();
  //Image -> trw();
  //Image -> trw(true);

  return 0;
}
