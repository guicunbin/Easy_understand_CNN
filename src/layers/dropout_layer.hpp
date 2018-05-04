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
        vector<Dtype> keep_mask;
        Dtype dropout_ratio_;



        void Setup(const LayerParameter<Dtype>& param,  MD_Vec<Dtype> *bottom,  MD_Vec<Dtype> *top){
            top->init_by_shape_and_constant(bottom->shape);
            keep_mask.resize(bottom->count(), 0);
            dropout_ratio_ = param.dropout_ratio;
        }


        conv_layer_class(const LayerParameter<Dtype>& param,  MD_Vec<Dtype> *bottom,  MD_Vec<Dtype> *top){
            Setup(param, bottom, top);
        }





        void Forward(MD_Vec<Dtype> *bottom,  MD_Vec<Dtype> *top, bool is_training = true){
            if(is_training){
                get_bernoulli_vec<Dtype>(bottom->count(), 1 - dropout_ratio,  &keep_mask.front());
            }
            compute_dropout(bottom, top, dropout_ratio_, &keep_mask.front(), is_training);
        }




        void Backward(MD_Vec<Dtype> *bottom,  MD_Vec<Dtype> *top, bool is_training = true){
            compute_dropout_backward(bottom, top, dropout_ratio_, &keep_mask.front(), is_training);
        }
};


#endif
