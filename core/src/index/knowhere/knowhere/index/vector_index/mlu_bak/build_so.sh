#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/neuware/lib64
export NEUWARE_HOME=/usr/local/neuware/
export PATH=$NEUWARE_HOME/bin:$PATH

g++ -shared -fPIC product_quantization_mlu.cc product_quantization.o topk.o -I . \
-I /usr/local/neuware/include/ -L /usr/local/neuware/lib64/ -lcnrt -o libmluIVFPQ.so