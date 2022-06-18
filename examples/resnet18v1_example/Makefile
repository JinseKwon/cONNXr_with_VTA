VTA_DIR = /home/xilinx/vta_lib
VTA_AR_DIR = -L$(VTA_DIR)/build
VTA_H_DIR = -I$(VTA_DIR)/include/zcu102 -I$(VTA_DIR)/3rdparty/dmlc-core/include

INC= -I ./include/
ONNX_INC= -I ../../include/ -I ../../protobuf/
LDLIBS= -fPIC ../../build/libconnxr.so -lm
LIB_CV= `pkg-config --libs opencv`

CFLAGS+= -DVTA $(VTA_H_DIR) -DVTA_COHERENT_ACCESSES=true 
LDLIBS+= -fopenmp -lvta_runtime -lcma -lpthread $(VTA_AR_DIR)

build:
    #rm -f runtest
    #gcc -std=c99 -Wall -D DEBUG -lm -O -c ../../src/operators/*.c ../../src/trace.c ../../src/utils.c ../../src/inference.c ../../protobuf/onnx.pb-c.c -I ../../include -I ../../protobuf
    #ar rcs libonnxcruntimelib.a *.o
	#ranlib libonnxcruntimelib.a
	g++  utils/image_io.c -c -o obj/image_io.o $(INC) $(LIB_CV) -lm
	g++  utils/class.c -c -o obj/class.o $(INC) -lm
	g++  utils/pbenc.c -c -o obj/pbenc.o $(INC) -lm
	g++  -W -Wall -g -O2 -o example obj/pbenc.o obj/image_io.o obj/class.o example.c $(INC) $(ONNX_INC) $(LDLIBS) $(LIB_CV) $(CFLAGS) -lm -DTRACE_LEVEL=1
	
clean:
	rm -f obj/*.o *.a *.gch
