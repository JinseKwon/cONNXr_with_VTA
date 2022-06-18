#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "class.h"

struct class_acc {
   float  acc;
   int    idx;
};
int compare_function(const void *a,const void *b) {
  float *x = (float *) a;
  float *y = (float *) b;
  if (*x > *y) return -1;
  else if (*x < *y) return 1; return 0;
}
void softmax(float *data,int n_class){
  float sum = 0.0;
  float max = -FLT_MAX;
  for(int i = 0;i < n_class; i++){
    if( data[i] > max ){
      max = data[i];
    }
  }
  for(int i = 0;i < n_class; i++){
    float e = expf(data[i] - max);
    sum += e;
    data[i] = e;
  }
  for(int i = 0;i < n_class; i++){
    data[i] = data[i] / sum;
  }
}
void classifier(float* data,int n_class){
  struct class_acc accuracy[n_class];
  
  softmax(data,n_class);

  for(int i = 0;i < n_class; i++){
    accuracy[i].acc = data[i];
    accuracy[i].idx = i;
  }
  qsort(accuracy, n_class, sizeof(float)+sizeof(int), compare_function);
  for(int i = 0;i < 5; i++){
    printf("class : %d (%.3f) : %s\n",accuracy[i].idx, accuracy[i].acc, class_name[accuracy[i].idx]);
  }
}

void class_sort(float* data,int n_class){
  struct class_acc accuracy[n_class];
  
  for(int i = 0;i < n_class; i++){
    accuracy[i].acc = data[i];
    accuracy[i].idx = i;
  }
  qsort(accuracy, n_class, sizeof(float)+sizeof(int), compare_function);
  for(int i = 0;i < 5; i++){
    printf("class : %d (%.3f) : %s\n",accuracy[i].idx, accuracy[i].acc, class_name[accuracy[i].idx]);
  }
}