#ifndef _FC_LAYER_HPP_
#define _FC_LAYER_HPP_


#include<vector>
#include"./LayerParameter.hpp"
#include"./layer.hpp"
#include"../utils/utils.hpp"
#include"../utils/base_utils.hpp"

template <typename Dtype>
class fc_layer_class //: public layer_class
{
    //private:

    public:

        typedef Dtype value_type;
        MD_Vec<Dtype> _kernel;
        MD_Vec<Dtype> _bias  ;
        MD_Vec<Dtype> _bias_multiplier  ;


        int BS;
        int C_in;
        int C_out;
        bool bias_term_ = false;


        fc_layer_class(const LayerParameter<Dtype>& param,  MD_Vec<Dtype> *bottom,  MD_Vec<Dtype> *top){
            //CHECK_equal(int(bottom->shape.size()),  2, any2str(__FILE__)+":"+any2str(__LINE__));
            BS    = bottom->shape[0];
            C_in  = bottom->count(1);
            C_out = param.out_channels;
            bias_term_ = param.bias_term;

            // init top 
            top->init_by_shape_and_constant({BS, C_out});
            
            // init _kernel
            _kernel.init_by_shape_and_constant({C_out,  C_in},  0);
            _kernel.data  = get_rand_vec<Dtype>(C_out * C_in, 0 ,param.weight_filler_std);
            _kernel.learn_rate = param.lr_w;




            if(bias_term_){
                //  //throw_error("the layer with bias are not implement");
                //  init bias
                _bias.init_by_shape_and_constant({C_out}, 0);
                _bias.data = get_rand_vec<Dtype>(_bias.count(), param.bias_filler_constant_value, 0);
                _bias.learn_rate = param.lr_b;
                _bias_multiplier.init_by_shape_and_constant({1}, 1);
            }
        }



        void Forward(MD_Vec<Dtype> *bottom,     MD_Vec<Dtype> *top){

            compute_fc_by_mat_mul(bottom, top, &_kernel);
            if(bias_term_){
                forward_cpu_bias(top,   &_bias,   &_bias_multiplier);   
            }
        }




        void Backward(MD_Vec<Dtype> *bottom,    MD_Vec<Dtype> *top){
            compute_fc_by_mat_mul_backward(bottom, top, &_kernel);
            _kernel.Update();
            if(bias_term_){
                backward_cpu_bias(top,   &_bias,   &_bias_multiplier);
                _bias.Update();
            }
        }
};


#endif

