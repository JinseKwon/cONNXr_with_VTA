#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../trace.h"
#include "operators.h"

int operator_quantizelinear(size_t n_input,
                            Onnx__TensorProto **input,
                            size_t n_attribute,
                            Onnx__AttributeProto **attribute,
                            size_t n_output,
                            Onnx__TensorProto **output)
{
  TRACE_LEVEL0("Calling operator_quantizelinear\n");
  if (0){
    /* TODO: Check some conditions. For example if a specific
     * functionality is not supported */
    return 1;
  }

  output[0]->dims   = malloc(input[0]->n_dims * sizeof(int64_t));
  output[0]->n_dims = input[0]->n_dims;

  for (int i = 0; i < output[0]->n_dims; i++)
  {
    output[0]->dims[i] = input[0]->dims[i];
  }
  output[0]->has_raw_data = 0;

  /* TODO hardcoded to uint8 */
  output[0]->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__UINT8;

  output[0]->n_int32_data = input[0]->n_float_data;
  output[0]->int32_data = malloc(output[0]->n_int32_data * sizeof(int32_t));
  /* TODO Only FLOAT is handled*/
  if (input[0]->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT){
    /* TODO third parameter is options, its assumed its always there */
    if (n_input != 3){return 1;}
    for(int i = 0; i < output[0]->n_int32_data; i++){
      output[0]->int32_data[i] = (input[0]->float_data[i] / input[1]->float_data[0]) + input[2]->int32_data[0];
    }
  }else{
    printf("wrong type %d\n", input[0]->data_type);
    return 1;
  }

  return 0;
}