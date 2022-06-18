#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

uint8_t* pb_formatting(uint8_t *input,int val, int len, int offset){
    
    for(int i =0; i<len-offset; i++){
      input[i+offset] = ((uint8_t)((val>>(i*7))<<1)>>1);
      if(i!=len-offset-1) input[i+offset] |= 128;
    }
    return input;
}
uint8_t* int8_2_pb(int val, int* len, int flag_on){
  uint8_t *result;
  int length = 0;
  if(flag_on > 0) flag_on = 1;
  length += flag_on;
  //0x08 == delimiter

  unsigned int val_test = val;
  for(int i = 0; i<5; i++){
    val_test = val_test >> 7;
    if(!val_test){
      length += i+1;
      break;
    }
  }
  result = (uint8_t *)malloc(length);
  if(flag_on) result[0] = 0x08;
  result = pb_formatting(result, val, length, flag_on);
  // for(int i =0; i < length; i++) printf("[%2x]",result[i]);
  *len = length;
  return result;
}

extern uint8_t* dim_2_pb(uint8_t *input,
                  int batch,int channel,int width,int height, 
                  int* length){
  uint8_t *result;

  uint8_t *b_data;  int b_len = 0;
  uint8_t *c_data;  int c_len = 0;
  uint8_t *w_data;  int w_len = 0;
  uint8_t *h_data;  int h_len = 0;
  uint8_t *f_data;  int f_len = 0;
  
  b_data = int8_2_pb(batch,   &b_len,1);
  c_data = int8_2_pb(channel, &c_len,1);
  w_data = int8_2_pb(width,   &w_len,1);
  h_data = int8_2_pb(height,  &h_len,1);
  int file_size = channel * width * height * 4;
  f_data = int8_2_pb(file_size,  &f_len,0);
  // printf("dim2pb\n");
  fflush(0);
  uint8_t delim[3] = {0x10, 0x01, 0x4a};

  int len = b_len + c_len + w_len + h_len + 3 + f_len;
  result = (uint8_t *)malloc(len);
  int offset = 0;
  for(int i=0; i<b_len; i++)  result[offset+i] = b_data[i];
  offset += b_len;
  for(int i=0; i<c_len; i++)  result[offset+i] = c_data[i];
  offset += c_len;
  for(int i=0; i<w_len; i++)  result[offset+i] = w_data[i];
  offset += w_len;
  for(int i=0; i<h_len; i++)  result[offset+i] = h_data[i];
  offset += h_len;
  for(int i=0; i<3; i++)  result[offset+i] = delim[i];
  offset += 3;
  for(int i=0; i<f_len; i++)  result[offset+i] = f_data[i];
  offset += f_len;
  
  // for(int i =0; i < len; i++) printf("[%2x]",result[i]);

  free(b_data);
  free(c_data);
  free(w_data);
  free(h_data);
  free(f_data);

  *length = len;
  return result;
}