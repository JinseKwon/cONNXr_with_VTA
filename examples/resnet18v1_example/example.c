#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "float.h"
#include <math.h>

#include "image_io.h"
// #include "pbenc.h"
// #include "class.h"

#include "onnx.pb-c.h"
#include "pbenc.h"
#include "utils.h"
#include "tracing.h"
#include "inference.h"
#include "operators/operator_set.h"

extern void classifier(float* data,int n_class);
extern void class_sort(float* data,int n_class);

struct layer{
  int n_w;
  float* weights;
  int n_b;
  float* bias;
  int has_bn;
  float* scale;
  float* mean;
  float* var;
  int l_dim[4];
};

#define INPUTIMG  "../../test/mobilenetv2-1.0/test_data_set_0/input_0.pb" 
#define OUTPUTIMG "../../test/mobilenetv2-1.0/test_data_set_0/output_0.pb"

#define INPUTMODEL "../../test/resnet18-v1-7/resnet18-v1-7.onnx"
// #define INPUTMODEL "./resnet18-v2-7.onnx"
// #define INPUTIMG   "../../test/tiny_yolov2/test_data_set_2/input_0.pb" 
// #define INPUTMODEL "./resnet-18.onnx"
// #define INPUTMODEL "./resnet18v2.onnx"
// #define INPUTMODEL "./resnet18-v2-7.onnx"
// #define INPUTMODEL "../Satrec_210224_model.onnx"
// #define INPUTMODEL "../../test/mobilenetv2-1.0/mobilenetv2-1.0.onnx"

/* TODO: Include statically linked library*/
Onnx__TensorProto *openTensorTEST_IMG(){
  Onnx__TensorProto *model = NULL;

  long len = 224*224*3;
  uint8_t *ret = (uint8_t*)calloc(len,1);

  model = onnx__tensor_proto__unpack(NULL,len,ret);

  return model;
}
Onnx__TensorProto *openTensorProtoFile2(char *fname){
  Onnx__TensorProto *model = NULL;

  TRACE(1, true, "Opening .pb file %s", fname);

  FILE *fl = fopen(fname, "r");
  if (fl == NULL){
    TRACE_ERROR(0, true, "File was not opened");
    TRACE_EXIT(1);
    return model;
  }

  fseek(fl, 0, SEEK_END);
  long len = ftell(fl);
  uint8_t *ret = (uint8_t*)malloc(len);
  fseek(fl, 0, SEEK_SET);
  for(long read = 0; read < len; read += fread(ret, 1, len-read, fl));
  fclose(fl);
  
  uint8_t *a = ret;
  // for(int i =0; i<13; i++)
  //   printf("ret[0] : %d\n",(uint8_t)ret[i]);

  TRACE(1, true, "length of file %ld", len);

  model = onnx__tensor_proto__unpack(NULL,len,ret);

  return model;
}
void batch_norm_l1(float* bias, float* scale, float* mean, float* var, 
                float* weights,
                int N,int C,int H, int W){
    //bias  K bias
    //gamma K scale
    //mean  K
    //var   K
    const float epsilon = 1e-5;
    
    for(int i=0; i<N; i++){
        //scale = gamma[i] / sqrt(var[i] + epsilon)
        float scalef = scale[i] / sqrt(var[i] + epsilon);
        // float scalef = 1.f / sqrt(var[i] + epsilon);
        bias[i] = bias[i] - mean[i] * scalef * 255.f;
        for(int w=0; w<C*H*W; ++w){
            weights[i*C*H*W + w] = weights[i*C*H*W + w] * scalef *255.f *255.f;
        }
    }
}
void batch_norm(float* bias, float* scale, float* mean, float* var, 
                float* weights,
                int N,int C,int H, int W){
    //bias  K bias
    //gamma K scale
    //mean  K
    //var   K
    const float epsilon = 1e-5;
    
    for(int i=0; i<N; i++){
        //scale = gamma[i] / sqrt(var[i] + epsilon)
        float scalef = scale[i] / sqrt(var[i] + epsilon);
        // float scalef = 1.f / sqrt(var[i] + epsilon);
        bias[i] = bias[i] - mean[i] * scalef;
        for(int w=0; w<C*H*W; ++w){
            weights[i*C*H*W + w] = weights[i*C*H*W + w] * scalef;
        }
    }
}
int main()
{
  /* Not working yet. Makefile need some love */
  
  Onnx__ModelProto *model = openOnnxFile(INPUTMODEL);
  if (model == NULL)
  {
    perror("Error when opening the onnx file\n");
    exit(-1);
  }

  /* TODO: Run some inference on MNIST examples */
  // Onnx__TensorProto *inp0set0 = openTensorProtoFile2(INPUTIMG);
  // Onnx__TensorProto *inp0set0 = openTensorTEST_IMG();
  // Onnx__TensorProto *out0set0 = openTensorProtoFile2(OUTPUTIMG);

  // Debug_PrintModelInformation(model);
  // convertRawDataOfTensorProto(out0set0);
  // Debug_PrintTensorProto(inp0set0);

  // convertRawDataOfTensorProto(inp0set0);

  Onnx__TensorProto *inp0set0;

  uint8_t *header;
  int header_len;
  int batch   = 1;
  int channel = 3;
  int width   = 256;
  int height  = 256;

  header = dim_2_pb(header, batch, channel, width, height, &header_len);
  // for(int i =0; i < header_len; i++) printf("[%2x]",header[i]);
  float* image_data = (float*)malloc(width*height*channel*sizeof(float));

  float range = 255.f;
  float m_r   = 0.485f;
  float m_g   = 0.456f;
  float m_b   = 0.406f;
  float std_r = 0.229f;
  float std_g = 0.224f;
  float std_b = 0.225f;

  // float range = 1.f;//255.f;
  // float m_r   = 104.0069879317889;//0.485f;
  // float m_g   = 116.66876761696767;//0.456f;
  // float m_b   = 122.6789143406786;//0.406f;
  // float std_r = 1.f;//0.229f;
  // float std_g = 1.f;//0.224f;
  // float std_b = 1.f;//0.225f;

  image_read_view("images/cat.jpg",image_data,width,
                  range, 
                  m_r,   m_g,   m_b, 
                  std_r, std_g, std_b);
  
  // image_read_view("images/dog.jpg",image_data,width,0.0f,0.0f,0.0f);
  uint8_t *data = (uint8_t *)calloc(width*height*channel*4+header_len,1);
  for(int i = 0; i<header_len; i++){
    data[i] = header[i];
  }

  inp0set0 = onnx__tensor_proto__unpack(NULL, width*height*channel*4 + header_len, data);
  convertRawDataOfTensorProto(inp0set0);
  for(int i = 0; i < channel*height*width; i++){
    inp0set0->float_data[i] = image_data[i];
  }
  free(image_data);
  // Debug_PrintTensorProto(inp0set0);
  // inp0set0->n_dims = 4;
  // inp0set0->dims[0] = 1;
  // inp0set0->dims[1] = 3;
  // inp0set0->dims[2] = 224;
  // inp0set0->dims[3] = 224;
  // inp0set0->has_data_type = 1;
  // inp0set0->n_float_data = 224*224*3;
  // for(int i = 0; i<224*224*3; i++){
  //   inp0set0->float_data[i] = 0.0f;
  // }
  inp0set0->name = model->graph->input[0]->name;
  
  Onnx__TensorProto *inputs[] = { inp0set0 };
  
  printf("[[Allocation ]]\n");
  fflush(stdout);
  resolve(model, inputs, 1);

  printf("[[DONE:allocation]]\n");

  char tensor_proto_data_type_string[17][100]= 
  { "undef", "float", "uint8", "int8", "uint16",
    "int16", "int32", "int64", "string", "bool",
    "float16", "double", "uint32", "uint64", "cmplx64",
    "cmplx128", "bfloat16"
  };

  printf("[[Inference ]]\n");
  clock_t start, end;
  double cpu_time_used;
  start = clock();  
  Onnx__TensorProto **output = inference(model, inputs, 1);
  end = clock();

  // for(int i = 0; i < all_context[_populatedIdx].outputs[0]->n_float_data; i++){
  //  printf("n_float_data[%d] = %f(%f)\n", i, all_context[_populatedIdx].outputs[0]->float_data[i],out0set0->float_data[i]);
  // }

  // non-softmax layer 
  classifier(all_context[_populatedIdx].outputs[0]->float_data, 1000);
  
  // with-softmax layer
  // class_sort(all_context[_populatedIdx].outputs[0]->float_data, 1000);

  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Predicted in %f seconds\n", cpu_time_used);

  /* 11 is hardcoded, which is Plus214_Output_0 */
  // compareAlmostEqualTensorProto(output[11], out0set0);

            
            // printf("[[file:writing]]\n");

            // char filename[100] = "resnet18_test.weights";
            // FILE *fpp;
            // fpp = fopen("resnet18_test.weights", "wb");
            
            // int head[5] = {1,2,0,0,0};
            // // for(int i =0;i<5;i++){
            //   fwrite(head, sizeof(int), 5, fpp);
            // // }

            // int conv_numb = 0;
            // for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++){
            //   if(!strcmp(model->graph->node[nodeIdx]->op_type,"Conv")
            //     || !strcmp(model->graph->node[nodeIdx]->op_type,"Gemm")){
            //     conv_numb++;
            //   }
            // }
            // struct layer *layers;
            // layers = (struct layer*)malloc(conv_numb*sizeof(struct layer));
            
            // printf("[[number of conv layer : %d]]\n",conv_numb);

            // //GRPAH LEVEL
            // printf("number of node : %d\n", model->graph->n_node);
            // conv_numb = 0;
            // for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++){
            //   all_context[nodeIdx].onnx_node = model->graph->node[nodeIdx];
            //   //node Print
            //   printf("NODE[%4d] : %10s : (# of in) %d : optype: %s\n",
            //                       nodeIdx,
            //                       model->graph->node[nodeIdx]->name,
            //                       model->graph->node[nodeIdx]->n_input,
            //                       model->graph->node[nodeIdx]->op_type);
            //   if(!strcmp(model->graph->node[nodeIdx]->op_type,"Conv")
            //   ||!strcmp(model->graph->node[nodeIdx]->op_type,"Gemm")){
            //     int i = 1;      //weights layer number
            //     layers[conv_numb].n_w = all_context[nodeIdx].inputs[i]->n_float_data;
            //     for(int d = 0; d < all_context[nodeIdx].inputs[i]->n_dims; d++){
            //       // layers[conv_numb].num_w *= all_context[nodeIdx].inputs[i]->dims[d];
            //       layers[conv_numb].l_dim[d] = all_context[nodeIdx].inputs[i]->dims[d];
            //     }
            //     if(all_context[nodeIdx].inputs[i]->n_dims == 2){
            //       layers[conv_numb].l_dim[2] = 1;
            //       layers[conv_numb].l_dim[3] = 1;
            //     }
            //     layers[conv_numb].n_w = all_context[nodeIdx].inputs[i]->n_float_data;
            //     layers[conv_numb].weights = (float*)malloc(layers[conv_numb].n_w*sizeof(float));
            //     for(int idx = 0; idx < all_context[nodeIdx].inputs[i]->n_float_data; idx++){
            //         layers[conv_numb].weights[idx] = all_context[nodeIdx].inputs[i]->float_data[idx];
            //     }
            //     if( model->graph->node[nodeIdx]->n_input > 2 ){
            //       i = 2;
            //       layers[conv_numb].n_b = all_context[nodeIdx].inputs[i]->n_float_data;
            //       layers[conv_numb].bias = (float*)malloc(layers[conv_numb].n_b*sizeof(float));
            //       for(int idx = 0; idx < all_context[nodeIdx].inputs[i]->n_float_data; idx++){
            //         layers[conv_numb].bias[idx] = all_context[nodeIdx].inputs[i]->float_data[idx];
            //       }
            //       layers[conv_numb].has_bn = 0;
            //       conv_numb++;
            //     }
            //   }else if(!strcmp(model->graph->node[nodeIdx]->op_type,"BatchNormalization")){
            //     layers[conv_numb].has_bn = 1;
            //     layers[conv_numb].n_b = all_context[nodeIdx].inputs[2]->n_float_data;
            //     layers[conv_numb].scale = (float*)malloc(layers[conv_numb].n_b*sizeof(float));
            //     layers[conv_numb].bias  = (float*)malloc(layers[conv_numb].n_b*sizeof(float));
            //     layers[conv_numb].mean  = (float*)malloc(layers[conv_numb].n_b*sizeof(float));
            //     layers[conv_numb].var   = (float*)malloc(layers[conv_numb].n_b*sizeof(float));
            //     for(int idx = 0; idx < all_context[nodeIdx].inputs[2]->n_float_data; idx++){
            //       // layers[conv_numb].scale[idx] = all_context[nodeIdx].inputs[1]->float_data[idx];
            //       // layers[conv_numb].bias[idx]  = all_context[nodeIdx].inputs[2]->float_data[idx];
            //       // layers[conv_numb].mean[idx]  = all_context[nodeIdx].inputs[3]->float_data[idx];
            //       // layers[conv_numb].var[idx]   = all_context[nodeIdx].inputs[4]->float_data[idx];
            //       //scale_weights    : scale all_context[nodeIdx].inputs[1];
            //       //scale_bias       : bias  all_context[nodeIdx].inputs[2];
            //       //bn_gamma/bn_mean : mean  all_context[nodeIdx].inputs[3];
            //       //bn_beta /bn_mean : var   all_context[nodeIdx].inputs[4];
                  
            //       layers[conv_numb].mean[idx]  = all_context[nodeIdx].inputs[3]->float_data[idx];
            //       layers[conv_numb].var[idx]   = all_context[nodeIdx].inputs[4]->float_data[idx];
            //       layers[conv_numb].scale[idx] = all_context[nodeIdx].inputs[1]->float_data[idx];
            //       layers[conv_numb].bias[idx]  = all_context[nodeIdx].inputs[2]->float_data[idx];
            //       // float tmp_mean  = layers[conv_numb].mean[idx];
            //       // float tmp_var   = layers[conv_numb].var[idx];
            //       // float tmp_scale = layers[conv_numb].scale[idx];
            //       // layers[conv_numb].bias[idx]  += (-tmp_mean)/sqrt(tmp_var + 0.00001f);
            //       // layers[conv_numb].scale[idx] =  (tmp_scale)/sqrt(tmp_var + 0.00001f);
            //       // layers[conv_numb].mean[idx]  = 0.0f;
            //       // layers[conv_numb].var[idx]   = 1.0f;
            //       // float bias_tmp = all_context[nodeIdx].inputs[2]->float_data[idx] - layers[conv_numb].mean[idx]/
            //       //                  sqrt(layers[conv_numb].mean[idx] + 0.00001f);
            //       // layers[conv_numb].bias[idx]  = bias_tmp;

            //     }
            //     conv_numb++;
            //   }else if(!strcmp(model->graph->node[nodeIdx]->op_type,"MaxPool")){

            //   }else if(!strcmp(model->graph->node[nodeIdx]->op_type,"Add")){

            //   }else if(!strcmp(model->graph->node[nodeIdx]->op_type,"AveragePool")){

            //   }else if(!strcmp(model->graph->node[nodeIdx]->op_type,"AveragePool")){

            //   }
            //   fflush(stdout);
            // }
            // printf("[[number of wroted conv layer : %d]]\n",conv_numb);

            // for(int i = 0; i<conv_numb; i++){
            //   printf("[l:%3d ] w:%10d b:%10d has_bn:%d  %4dx%4dx%4dx%4d\n",
            //     i,layers[i].n_w,layers[i].n_b,layers[i].has_bn,
            //     layers[i].l_dim[0],layers[i].l_dim[1],layers[i].l_dim[2],layers[i].l_dim[3]);

            //   //covolution weights write;
              
            //   if(layers[i].has_bn){
            //     //bias/scale/mean/var write;
            //     printf("bias  :");
            //     for(int j=0; j<10; j++){
            //       printf("%.4f ",layers[i].bias[j]);
            //     }
            //     printf("\nscale :");
            //     for(int j=0; j<10; j++){
            //       printf("%.4f ",layers[i].scale[j]);
            //     }
            //     printf("\nmean  :");
            //     for(int j=0; j<10; j++){
            //       printf("%.4f ",layers[i].mean[j]);
            //     }
            //     printf("\nvar   :");
            //     for(int j=0; j<10; j++){
            //       printf("%.4f ",layers[i].var[j]);
            //     }
            //     printf("\nweight:");
            //     for(int j=0; j<10; j++){
            //       printf("%.4f ",layers[i].weights[j]);
            //     }
            //     // printf("\n");
            //     // if(i == 0)
            //     //   for(int j=0; j<layers[i].n_w; j++)
            //     //     printf("%.4f ",layers[i].weights[j]);
            //     printf("\n");
            //     // batch_norm( layers[i].bias,
            //     //             layers[i].scale,
            //     //             layers[i].mean,
            //     //             layers[i].var,
            //     //             layers[i].weights,
            //     //             layers[i].l_dim[0], 
            //     //             layers[i].l_dim[1], 
            //     //             layers[i].l_dim[2],
            //     //             layers[i].l_dim[3]);
            //     fwrite(layers[i].bias,    sizeof(float),  layers[i].n_b,  fpp);
            //     fwrite(layers[i].scale,   sizeof(float),  layers[i].n_b,  fpp);
            //     fwrite(layers[i].mean,    sizeof(float),  layers[i].n_b,  fpp);
            //     fwrite(layers[i].var,     sizeof(float),  layers[i].n_b,  fpp);
            //     fwrite(layers[i].weights, sizeof(float),  layers[i].n_w,  fpp);

            //   }else{
            //           printf("bias  :");
            //     for(int j=0; j<10; j++){
            //       printf("%.4f ",layers[i].bias[j]);
            //     }
            //     printf("\nweight:");
            //     for(int j=0; j<10; j++){
            //       printf("%.4f ",layers[i].weights[j]);
            //     }
            //     printf("\n");  
            //     //bias write;
            //     // for(int j=0; j<10; j++){
            //     //   printf("%.4f ",layers[i].bias[j]);
            //     // }
            //     fwrite(layers[i].bias, sizeof(float),layers[i].n_b,fpp);
            //     fwrite(layers[i].weights,sizeof(float),layers[i].n_w,fpp);
            //   }
            //   // printf("\n");
            //   // for(int j=0; j<10; j++){
            //   //   printf("%.4f ",layers[i].weights[j]);
            //   // }
            //   // printf("\n");
            // }
            // fclose(fpp);
            // printf("[[file wrote]]\n");


  /* TODO: Free all resources */
  free(data);
  
  return 0;
}
