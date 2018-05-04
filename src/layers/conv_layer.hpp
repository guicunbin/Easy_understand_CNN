#ifndef _CONV_LAYER_HPP_
#define _CONV_LAYER_HPP_


#include"./layer.hpp"
#include"./LayerParameter.hpp"
#include"../utils/utils.hpp"

template <typename Dtype>
class conv_layer_class //: public layer_class
{
    //private:

    public:

        typedef Dtype value_type;
        MD_Vec<Dtype> _kernel;
        MD_Vec<Dtype> _bias  ;
        MD_Vec<Dtype> _bias_multiplier;

        int kernel_size;
        int stride;
        int pad;
        int BS;
        int C_in,  H_in,    W_in;
        int C_out, H_out,   W_out;
        bool bias_term_ = false;


        void Setup(const LayerParameter<Dtype>& param,  MD_Vec<Dtype> *bottom,  MD_Vec<Dtype> *top){
            CHECK_equal(int(bottom->shape.size()),  4,  any2str(__FILE__)+":"+any2str(__LINE__));
            //cout<<"this is in "<<__FILE__<<endl;
            kernel_size = param.kernel_size;
            stride = param.stride;
            pad = param.pad;
            bias_term_ = param.bias_term;


            BS = bottom->shape[0];
            C_in  = bottom->shape[1];     
            H_in = bottom->shape[2];  
            W_in = bottom->shape[3];

            C_out = param.out_channels;
            H_out = (H_in - kernel_size + 2 * pad) / stride + 1;
            W_out = (W_in - kernel_size + 2 * pad) / stride + 1;

            //  cout<<__LINE__<<endl;
            // init top 
            top->init_by_shape_and_constant({BS, C_out, H_out, W_out},0);
            

            // init  _kernel
            _kernel.init_by_shape_and_constant({C_out, C_in, kernel_size,  kernel_size}, 0);
            //  cout<<__LINE__<<endl;
            _kernel.data  = get_rand_vec<Dtype>(_kernel.count(), 0, param.weight_filler_std);
            _kernel.learn_rate = param.lr_w;
            //  cout<<__LINE__<<endl;

            if(bias_term_){
                //  //throw_error("the layer with bias are not implement");
                //  init bias
                _bias.init_by_shape_and_constant({C_out}, 0);
                _bias.data = get_rand_vec<Dtype>(_bias.count(), param.bias_filler_constant_value, 0);
                _bias.learn_rate = param.lr_b;
                _bias_multiplier.init_by_shape_and_constant({H_out * W_out}, 1);
            }
        }


        conv_layer_class(const LayerParameter<Dtype>& param,  MD_Vec<Dtype> *bottom,  MD_Vec<Dtype> *top){
            Setup(param, bottom, top);
        }


        void Forward(MD_Vec<Dtype> *bottom,  MD_Vec<Dtype> *top){
            //compute_conv2d_by_7_for(bottom, top, weight_mat, stride, pad);
            // copy_to  will not change the buffer address
            //cout<<__FILE__<<":"<<__LINE__<<endl;

            compute_conv2d_by_mat_mul(bottom, top, &_kernel, stride, pad);
            if(bias_term_){
                forward_cpu_bias(top,   &_bias,   &_bias_multiplier);   
            }
        }




        void Backward(MD_Vec<Dtype> *bottom,  MD_Vec<Dtype> *top){
            compute_conv2d_by_mat_mul_backward(bottom, top, &_kernel, stride, pad);
            _kernel.Update();
            if(bias_term_){
                backward_cpu_bias(top,   &_bias,   &_bias_multiplier);
                _bias.Update();
            }
        }
};


#endif
