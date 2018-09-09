#   Easy_understand_CNN
##  Easy understand CNN Framwork Implemented from scratch by using C++

### Usage

- Test utils functions
> cd ./test/test_utils/ && ./compile_and_run.sh

- Get data
> ./data/mnist/get_mnist.sh

- Training and testing    
> cd ./example/mnist/ && ./compile_and_run.sh


### Defining and training your own model architecture
#### see the ./example/mnist/main.cpp 
- define the layer param
> const LayerParameter<double> conv1_param(LayerType::conv,  20,  true, lr, lr, filler::gaussian, filler::constant, 0.01, 0.0, 7, 7,0)
- define the layer class
> conv_layer_class<double> Conv1(conv1_param, &bottom, &tops[0]);
- training
> Conv1.Forward(&bottom,      &tops[idx]);    idx++;
> Conv1.Backward(&bottom,     &tops[idx]);    idx--;
- testing
> Conv1.Forward(&bottom,      &tops[idx]);    idx++;

