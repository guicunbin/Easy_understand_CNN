#ifndef _RELU_LAYER_HPP_
#define _RELU_LAYER_HPP_


#include"./layer.hpp"
#include"./LayerParameter.hpp"
#include"../utils/utils.hpp"

template <typename Dtype>
class relu_layer_class //: public layer_class
{
    //private:

    public:

        typedef Dtype value_type;

        relu_layer_class(const LayerParameter<Dtype>& param,  MD_Vec<Dtype> *bottom,  MD_Vec<Dtype> *top){
            // init top 
            // because top == bottom
            CHECK_vector_equal(bottom->shape,   top->shape);
            //top->init_by_shape_and_constant(bottom.shape);
        }





        void Forward(MD_Vec<Dtype> *bottom,     MD_Vec<Dtype> *top){
            math_RELU(bottom);
            copy_vector_from_to(bottom->data, top->data, 0, bottom->count()); 
        }




        void Backward(MD_Vec<Dtype> *bottom,    MD_Vec<Dtype> *top){
            math_RELU_backward(top);
            copy_vector_from_to(top->diff,  bottom->diff, 0, bottom->count());
        }
};


#endif

