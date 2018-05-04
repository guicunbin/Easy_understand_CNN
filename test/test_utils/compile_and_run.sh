 #g++ test_utils.cpp --std=c++11
 g++ test_utils.cpp --std=c++11 -lcblas `pkg-config --cflags --libs opencv` -o Test_O && ./Test_O
 #g++ test_utils.cpp --std=c++11 -lcblas
