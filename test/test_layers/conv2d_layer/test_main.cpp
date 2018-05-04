#include"../../../src/layers/conv_layer.hpp"
#include"../../../src/utils/utils.hpp"


int main()
{
    MD_Vec<double>  bottom({1, 3, 16, 16});
    vector<double> X = get_rand_vec<double>(bottom.count(), 1, 2);
    copy_vector_from_to(X, bottom.data, 0, bottom.count());
    MD_Vec<double>  top;

    // LayerType,  out_channels,    bias_term_,   lr_w,     lr_b,     
    // filler::w, filler::b, w_std, b_constant,     
    // kernel_size,    stride,     pad
    const LayerParameter<double> conv1_param(LayerType::conv, 1, false, 1, 0.1, filler::gaussian, filler::constant, 0.01, 0.0, 5, 5, 0); 
    
    //conv1_param.print();

    conv_layer_class<double> conv1(conv1_param, &bottom, &top);

    //cout<<" exit()"<<endl; exit(0);

    //cout<<"start Forward  "<<endl;
    clock_t start = clock();
    conv1.Forward(&bottom, &top);
    //print_time_using(start, "conv1.Forward using time = ");


    print_vec(conv1._kernel.data,   "weight = \\");

    print_vec(bottom.data,          "bottom = \\");
    
    print_vec(top.data,             "top = \\");

    return 0;
}

