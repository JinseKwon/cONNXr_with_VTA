#!/bin/bash

make -j 4 all
cd examples/resnet18v1_example
./example
cd ../../
