#g++ main.cpp --std=c++11 && ./a.out #> ./data.py && \
# CUDA_VISIBLE_DEVICES=1 python ./run.py

g++ ./test_main.cpp -o ./TEST_MAIN_CPP --std=c++11 -lcblas `pkg-config --cflags`  && ./TEST_MAIN_CPP > ./test_data.py && \
CUDA_VISIBLE_DEVICES=1 python ./test_conv2d_use_tf.py  && \
rm -rf ./test_data.py*
