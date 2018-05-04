#ifndef _LAYERPARAMETER_HPP_
#define _LAYERPARAMETER_HPP_

#include"../utils/utils.hpp"
using namespace std;



template <typename Dtype>
class LayerParameter{
    public:
        typedef Dtype value_type;
        int kernel_size, stride, pad;
        int out_channels;
        bool bias_term;
        Dtype lr_w, lr_b;
        LayerType layer_type;
        filler weight_filler,   bias_filler;
        Dtype weight_filler_std;
        Dtype bias_filler_constant_value;
        Dtype dropout_ratio;




        LayerParameter(LayerType layer_type_ = LayerType::conv,
                        int out_channels_ = 3 ,
                        bool bias_term_  = false,
                        Dtype lr_w_ = 1, Dtype lr_b_ = 0.1, 
                        filler weight_filler_ = filler::gaussian, 
                        filler bias_filler_ = filler::constant,
                        Dtype weight_filler_std_ = 0.01, 
                        Dtype bias_filler_constant_value_ = 0.0,
                        int kernel_size_ = 2, int stride_ = 2, int pad_ = 0,
                        bool dropout_ratio_ = 0.5){

            layer_type = layer_type_;    
            bias_term  = bias_term_;
            out_channels = out_channels_;
            kernel_size = kernel_size_;
            stride = stride_;    pad = pad_;  lr_w = lr_w_;    lr_b = lr_b_;
            weight_filler = weight_filler_;  bias_filler = bias_filler_;
            weight_filler_std = weight_filler_std_;
            bias_filler_constant_value   = bias_filler_constant_value_;
            dropout_ratio  = dropout_ratio_;
            }


        void print() const {
            string layer_type_str = layer_type == LayerType::conv ? "conv" : "fc";
            string weight_filler_str = weight_filler == filler::gaussian ? "gaussian" : "constant";
            string bias_filler_str = bias_filler == filler::gaussian ? "gaussian" : "constant";
            cout<<"{---------------------------------------------------------"<<endl;
            cout<<"LayerType:                   "<<layer_type_str<<endl;
            cout<<"out_channels:                "<<out_channels<<endl;
            cout<<"kernel_size:                 "<<kernel_size<<endl;
            cout<<"stride:                      "<<stride<<endl;
            cout<<"pad:                         "<<pad<<endl;
            cout<<"bias_term:                   "<<bias_term<<endl;
            cout<<"lr_w:                        "<<lr_w<<endl;
            cout<<"lr_b:                        "<<lr_b<<endl;
            cout<<"weight_filler:               "<<weight_filler_str<<endl;
            cout<<"bias_filler:                 "<<bias_filler_str<<endl;
            cout<<"weight_filler_std:           "<<weight_filler_std<<endl;
            cout<<"bias_filler_constant_value:  "<<bias_filler_constant_value<<endl;
            cout<<"---------------------------------------------------------}"<<endl;
        }
};



#endif
