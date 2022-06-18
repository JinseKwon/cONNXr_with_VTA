#include "opencv/highgui.h"

// extern "C";

// extern void image_read_view(char *img_file, float *image, int img_size,
//                      float mean_r,   float mean_g, float mean_b);
void image_read_view(char *img_file, float *image, int img_size,
                     float raw_255_to_1f,
                     float mean_r,   float mean_g, float mean_b,
                     float stdd_r,   float stdd_g, float stdd_b);
// IplImage* image_read(char *img_file, float *image, int img_size,
//                      float mean_r,   float mean_g, float mean_b);
