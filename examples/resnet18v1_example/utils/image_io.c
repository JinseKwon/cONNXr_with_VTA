#include "opencv/highgui.h"

#include <stdio.h>
#include "string.h"
#include <stdlib.h>

const char class_name[20][20] = {"plane", "bicycle", "bird", "boat", "bottle", 
                                "bus", "car", "cat", "chair", "cow", 
                                "diningtable", "dog", "horse", "motorbike", "person", 
                                "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
const char color_map[20][3] ={  {255,255,255},{153,63,118},{171,4,255},{255,251,141},{204,179,23},
                            {155,204,39},{250,255,0},{255,65,0},{137,255,196},{20,204,60},
                            {204,172,23},{0,255,134},{187,204,180},{66,214,255},{255,228,23},
                            {204,150,192},{153,63,134},{148,4,255},{255,242,66},{204,173,23}};
float scale;

void image_read_view(char *img_file, float *image, int img_size,
                     float raw_255_to_1f,
                     float mean_r,   float mean_g, float mean_b,
                     float stdd_r,   float stdd_g, float stdd_b){
    float scale;
    //  input_img = cvLoadImage("images/dog.jpg");    
    IplImage *input_img = cvLoadImage(img_file); 
    if(!input_img){
        printf("Can't load image\n");
        exit(0);
    }
    IplImage *readimg = cvCreateImage(cvSize(img_size,img_size),IPL_DEPTH_8U,3);
    //  input_img = cvLoadImage("images/black_dog.jpg");  
    // input_img = cvLoadImage("images/horses.jpg");    
    int x = input_img -> width;
    int y = input_img -> height;
    // //printf("h,w : %d %d\n",y, x);
    if(x > y){
        int new_x = (x-y)/2;
        scale = (float)y / (float)img_size;
        cvSetImageROI(input_img,cvRect(new_x,0,x-new_x*2,y));
       // printf("(x0,y0,w,h) : %d %d, %d %d\n",new_x,0,x-new_x*2,y);
        // cvSetImageROI(input_img,cvRect(0,0,x,x));
        
    }else if(y > x){
        int new_y = (y-x)/2;
        scale = (float)x / (float)img_size;
        cvSetImageROI(input_img,cvRect(0,new_y,x,y-new_y));
       // printf("(x0,y0,w,h) : %d %d, %d %d\n",0,new_y,x,y-new_y*2);
    }
    cvResize(input_img, readimg, 0);
    // char *imgs = (char*)malloc(416 * 416 * 3);
    unsigned char *imgs = (unsigned char*)readimg->imageData;
    
    //opencv 4 times padding for speed up
    int img_padding = readimg->widthStep;
    
    for(int h =0; h<img_size; ++h){
        for(int w =0; w<img_size; ++w){
            // for(int c = 0; c<3; c++){
            //     image[c*img_size*img_size + h*img_size + w] = imgs[ h*img_padding + w*3 + c] / 255.;
            image[0*img_size*img_size + h*img_size + w] = (imgs[ h*img_padding + w*3 + 0] / raw_255_to_1f - mean_r)/stdd_r;
            image[1*img_size*img_size + h*img_size + w] = (imgs[ h*img_padding + w*3 + 1] / raw_255_to_1f - mean_g)/stdd_g;
            image[2*img_size*img_size + h*img_size + w] = (imgs[ h*img_padding + w*3 + 2] / raw_255_to_1f - mean_b)/stdd_b;
            // }
        }
    }
    // cvSaveImage("input.jpg", input_img);
    // free(imgs);
    // for(int kk = 0; kk < img_size * img_size * 3; ++kk){
    //     readimg->imageData[kk] = (char)(image[kk] * 255 );
    // }
    // cvShowImage("iamge show",readimg);
    // cvWaitKey(1000);
    cvReleaseImage(&input_img);
}
/*
IplImage* image_read(char *img_file, float *image, int img_size,
                     float mean_r,   float mean_g, float mean_b){
                         
    //  input_img = cvLoadImage("images/dog.jpg");    
    IplImage *input_img = cvLoadImage(img_file); 
    if(!input_img){
        printf("Can't load image\n");
        exit(0);
    }
    IplImage *readimg = cvCreateImage(cvSize(img_size,img_size),IPL_DEPTH_8U,3);
    //  input_img = cvLoadImage("images/black_dog.jpg");  
    // input_img = cvLoadImage("images/horses.jpg");    
    int x = input_img -> width;
    int y = input_img -> height;
    // //printf("h,w : %d %d\n",y, x);
    if(x > y){
        int new_x = (x-y)/2;
        scale = (float)y / (float)img_size;
        cvSetImageROI(input_img,cvRect(new_x,0,x-new_x*2,y));
       // printf("(x0,y0,w,h) : %d %d, %d %d\n",new_x,0,x-new_x*2,y);
        // cvSetImageROI(input_img,cvRect(0,0,x,x));
        
    }else if(y > x){
        int new_y = (y-x)/2;
        scale = (float)x / (float)img_size;
        cvSetImageROI(input_img,cvRect(0,new_y,x,y-new_y));
       // printf("(x0,y0,w,h) : %d %d, %d %d\n",0,new_y,x,y-new_y*2);
    }
    cvResize(input_img, readimg, 0);
    // char *imgs = (char*)malloc(416 * 416 * 3);
    unsigned char *imgs = (unsigned char*)readimg->imageData;
    
    //opencv 4 times padding for speed up
    int img_padding = readimg->widthStep;
    
    for(int h =0; h<img_size; ++h){
        for(int w =0; w<img_size; ++w){
            // for(int c = 0; c<3; c++){
            //     image[c*img_size*img_size + h*img_size + w] = imgs[ h*img_padding + w*3 + c] / 255.;
            image[0*img_size*img_size + h*img_size + w] = (imgs[ h*img_padding + w*3 + 0]-mean_r)/ 255.;
            image[1*img_size*img_size + h*img_size + w] = (imgs[ h*img_padding + w*3 + 1]-mean_g)/ 255.;
            image[2*img_size*img_size + h*img_size + w] = (imgs[ h*img_padding + w*3 + 2]-mean_b)/ 255.;
            // }
        }
    }
    // cvSaveImage("input.jpg", input_img);
    // free(imgs);
    // for(int kk = 0; kk < img_size * img_size * 3; ++kk){
    //     readimg->imageData[kk] = (char)(image[kk] * 255 );
    // }
    // cvShowImage("YOLO tiny with OpenCL",readimg);
    // cvWaitKey(33);
    // cvReleaseImage(&input_img);
    return input_img;
}
IplImage* Ipl_read(IplImage* input_img, float *image, int img_size,
                     float mean_r,   float mean_g, float mean_b){   
    
    IplImage *readimg = cvCreateImage(cvSize(img_size,img_size),IPL_DEPTH_8U,3);
    int x = input_img -> width;
    int y = input_img -> height;
    // //printf("h,w : %d %d\n",y, x);
    if(x > y){
        int new_x = (x-y)/2;
        scale = (float)y / (float)img_size;
        cvSetImageROI(input_img,cvRect(new_x,0,x-new_x*2,y));
       // printf("(x0,y0,w,h) : %d %d, %d %d\n",new_x,0,x-new_x*2,y);
        // cvSetImageROI(input_img,cvRect(0,0,x,x));
        
    }else if(y > x){
        int new_y = (y-x)/2;
        scale = (float)x / (float)img_size;
        cvSetImageROI(input_img,cvRect(0,new_y,x,y-new_y));
       // printf("(x0,y0,w,h) : %d %d, %d %d\n",0,new_y,x,y-new_y*2);
    }
    cvResize(input_img, readimg, 0);
    // char *imgs = (char*)malloc(416 * 416 * 3);
    unsigned char *imgs = (unsigned char*)readimg->imageData;
    
    //opencv 4 times padding for speed up
    int img_padding = readimg->widthStep;
    
    for(int h =0; h<img_size; ++h){
        for(int w =0; w<img_size; ++w){
            // for(int c = 0; c<3; c++){
            //     image[c*img_size*img_size + h*img_size + w] = imgs[ h*img_padding + w*3 + c] / 255.;
            image[0*img_size*img_size + h*img_size + w] = (imgs[ h*img_padding + w*3 + 0]-mean_r)/ 255.;
            image[1*img_size*img_size + h*img_size + w] = (imgs[ h*img_padding + w*3 + 1]-mean_g)/ 255.;
            image[2*img_size*img_size + h*img_size + w] = (imgs[ h*img_padding + w*3 + 2]-mean_b)/ 255.;
            // }
        }
    }
    // free(imgs);
    // for(int kk = 0; kk < img_size * img_size * 3; ++kk){
    //     readimg->imageData[kk] = (char)(image[kk] * 255 );
    // }
    // cvShowImage("YOLO tiny with OpenCL",readimg);
    // cvWaitKey(33);
    // cvReleaseImage(&input_img);
    return input_img;
}
void image_show_class(char* class_text, IplImage *readimg,
                      double elapsed, int show_time){
    char text[20];
    CvFont font, font_guide, font_txt;
        
    cvInitFont(&font, CV_FONT_HERSHEY_DUPLEX, 0.7, 0.7, 0 ,2);//, 0, 1, CV_AA);
    cvInitFont(&font_guide, CV_FONT_HERSHEY_DUPLEX, 1.0, 1.0, 0 ,3);//, 0, 1, CV_AA);
    cvInitFont(&font_txt, CV_FONT_HERSHEY_SIMPLEX, 0.6, 0.6, 0 ,2);//, 0, 1, CV_AA);

    // double elapsed = timer_stop(0);
    // double elapsed = 1;
    IplImage *newimg = cvCreateImage(cvSize(500,500),IPL_DEPTH_8U,3);
    cvResize(readimg,newimg);
    // classificaiton result print
    // printf("%s\n",class_text);
    char *ptr;
    ptr = strtok(class_text, "\n");
    for(int i=0; i<5; ++i){
        cvPutText(newimg, ptr, cvPoint(5,375+i*25), 
                  &font, CV_RGB(237,193,26));
        ptr = strtok(NULL,"\n");
    }
    sprintf(text, "%.1f", 1/elapsed);//1/(get_time()-end_time));
    cvPutText(newimg, text, cvPoint(350, 50), &font_guide, CV_RGB(4, 255, 75));    //FPS val
    cvPutText(newimg, "FPS", cvPoint(350, 20), &font_txt, CV_RGB(4, 255, 75));    //FPS txt

    // printf("Elapsed time  %.6f \n", elapsed);
    // printf("%s\n", text);
    sprintf(text, "%.0fms", 1000*elapsed);//1/(get_time()-end_time));
    cvPutText(newimg, "latency", cvPoint(200, 24), &font_txt, CV_RGB(255, 29, 29));    //latency txt
    cvPutText(newimg, text, cvPoint(200, 50), &font_guide, CV_RGB(255, 29, 29));    //latency val

    cvShowImage("Image Classification",newimg);
    // cvWaitKey(10);
    cvWaitKey(show_time);

    //if(cvWaitKey(33) == 1048691){   "s" key input
}
void image_show_yolo(float* box_output, 
                    int BOX_cnt,
                    IplImage *readimg,
                    double elapsed, int show_time){
    char text[20];
    CvFont font, font_guide, font_txt;

    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0 ,2);//, 0, 1, CV_AA);
    cvInitFont(&font_guide, CV_FONT_HERSHEY_DUPLEX, 1.0, 1.0, 0 ,3);//, 0, 1, CV_AA);
    cvInitFont(&font_txt, CV_FONT_HERSHEY_SIMPLEX, 0.6, 0.6, 0 ,2);//, 0, 1, CV_AA);

    // double elapsed = timer_stop(0);
    // double elapsed = 1;
    // printf("printing box... %d <<< \n",(int)box_output[13*13*5*20*6]);
    if((int)box_output[BOX_cnt*6] > 0){
       for(int i = 0; i < (int)box_output[BOX_cnt*6]; i++){
            sprintf(text, "%s(%.2f)", class_name[(int)box_output[i*6 + 1]], box_output[i*6] );//1/(get_time()-end_time));
            printf("%s\n", text);
            int class_i = (int)box_output[i*6 + 1];
            cvPutText(readimg, text, 
                      cvPoint((int)(box_output[i*6 + 2]*scale), 
                              (int)(box_output[i*6 + 3]*scale)), 
                      &font, 
                      CV_RGB(color_map[class_i][0],
                             color_map[class_i][1],
                             color_map[class_i][2])
                      );
            cvRectangle(readimg, 
                        cvPoint((int)(box_output[i*6 + 2]*scale), 
                                (int)(box_output[i*6 + 3]*scale)), 
                        cvPoint((int)(box_output[i*6 + 4]*scale), 
                                (int)(box_output[i*6 + 5]*scale)),
                        CV_RGB(color_map[class_i][0],
                               color_map[class_i][1],
                               color_map[class_i][2]),
                        2);
            printf("lt x,y %d,%d\n", box_output[i*6 + 2]*scale,box_output[i*6 + 3]*scale);
        }
    }
    sprintf(text, "%.1f", 1/elapsed);//1/(get_time()-end_time));
    cvPutText(readimg, text, cvPoint(350, 50), &font_guide, CV_RGB(4, 255, 75));    //FPS val
    cvPutText(readimg, "FPS", cvPoint(350, 20), &font_txt, CV_RGB(4, 255, 75));    //FPS txt

    // printf("Elapsed time  %.6f \n", elapsed);
    // printf("%s\n", text);
    sprintf(text, "%.0fms", 1000*elapsed);//1/(get_time()-end_time));
    cvPutText(readimg, "latency", cvPoint(200, 24), &font_txt, CV_RGB(255, 29, 29));    //latency txt
    cvPutText(readimg, text, cvPoint(200, 50), &font_guide, CV_RGB(255, 29, 29));    //latency val

    cvShowImage("Object Detection",readimg);
    cvWaitKey(show_time);
    cvSaveImage("saved.jpg", readimg);
    //if(cvWaitKey(33) == 1048691){   "s" key input
}
void image_show_yolo3(float* box_output, 
                    int BOX_cnt,
                    int n_class,
                    IplImage *readimg,
                    double elapsed, int show_time){
    char text[20];
    CvFont font, font_guide, font_txt;
    // printf("%d box count \n",BOX_cnt);
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0 ,2);//, 0, 1, CV_AA);
    cvInitFont(&font_guide, CV_FONT_HERSHEY_DUPLEX, 1.0, 1.0, 0 ,3);//, 0, 1, CV_AA);
    cvInitFont(&font_txt, CV_FONT_HERSHEY_SIMPLEX, 0.6, 0.6, 0 ,2);//, 0, 1, CV_AA);
    
    cvSaveImage("input.jpg", readimg);

    // double elapsed = timer_stop(0);
    // double elapsed = 1;
    // printf("printing box... %d <<< \n",(int)box_output[13*13*5*20*6]);
    if(BOX_cnt > 0){
       for(int i = 0; i < BOX_cnt; i++){
            if(box_output[i*(5+n_class)] == 0) continue;
            int class_i = 0;
            sprintf(text, "%s(%.2f)", 
                    class_name[class_i], //TODO...classname change
                    box_output[i*(5+n_class)] );//1/(get_time()-end_time));
            // printf("%s\n", text);
            cvPutText(readimg, text, 
                      cvPoint((int)(box_output[i*(5+n_class) + 1]*scale), 
                              (int)(box_output[i*(5+n_class) + 2]*scale)), 
                      &font, 
                      CV_RGB(color_map[class_i][0],
                             color_map[class_i][1],
                             color_map[class_i][2])
                      );
            cvRectangle(readimg, 
                        cvPoint((int)(box_output[i*(5+n_class) + 1]*scale), 
                                (int)(box_output[i*(5+n_class) + 2]*scale)), 
                        cvPoint((int)(box_output[i*(5+n_class) + 3]*scale), 
                                (int)(box_output[i*(5+n_class) + 4]*scale)),
                        CV_RGB(color_map[class_i][0],
                               color_map[class_i][1],
                               color_map[class_i][2]),
                        2);
            // printf("lt x,y %f,%f \n", box_output[i*6 + 2]*scale,box_output[i*(5+n_class) + 3]*scale);
            // printf("rb x,y %f,%f \n", box_output[i*6 + 4]*scale,box_output[i*(5+n_class) + 5]*scale);
        }
    }
    sprintf(text, "%.1f", 1/elapsed);//1/(get_time()-end_time));
    // cvPutText(readimg, text, cvPoint(350, 50), &font_guide, CV_RGB(4, 255, 75));    //FPS val
    // cvPutText(readimg, "FPS", cvPoint(350, 20), &font_txt, CV_RGB(4, 255, 75));    //FPS txt

    // printf("Elapsed time  %.6f \n", elapsed);
    // printf("%s\n", text);
    sprintf(text, "%.0fms", 1000*elapsed);//1/(get_time()-end_time));
    // cvPutText(readimg, "latency", cvPoint(200, 24), &font_txt, CV_RGB(255, 29, 29));    //latency txt
    // cvPutText(readimg, text, cvPoint(200, 50), &font_guide, CV_RGB(255, 29, 29));    //latency val
    
    IplImage *newimg = cvCreateImage(cvSize(500,500),IPL_DEPTH_8U,3);
    cvResize(readimg,newimg);
    
    cvShowImage("Object Detection",newimg);
    cvWaitKey(show_time);
    cvSaveImage("saved.jpg", readimg);
    //if(cvWaitKey(33) == 1048691){   "s" key input
}
void image_free(){
    // cvReleaseCapture(&capture);
    // cvReleaseImage(&readimg);
    cvDestroyWindow("YOLO tiny with OpenCL");
}
    
*/