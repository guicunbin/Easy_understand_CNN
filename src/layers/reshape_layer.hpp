#ifndef _RESHAPE_LAYER_HPP_
#define _RESHAPE_LAYER_HPP_


#include"./layer.hpp"
#include"./LayerParameter.hpp"
#include"../utils/utils.hpp"

template <typename Dtype>
class reshape_layer_class //: public layer_class
{
    //private:

    public:

        typedef Dtype value_type;



        reshape_layer_class(const LayerParameter<Dtype>& param,  MD_Vec<Dtype> *bottom,  MD_Vec<Dtype> *top){
            if(bottom->shape.size() == 4){
                vector<int> top_shape = {bottom->shape[0], bottom->count(1,4)};
                top->init_by_shape_and_constant(top_shape,  0);
            }
            else{
                throw_error("another reshape  not implement");
            }
        }





        void Forward(MD_Vec<Dtype> *bottom,  MD_Vec<Dtype> *top){
            for(int i=0; i<top->count();    i++){
                top->data[i] = bottom->data[i];
            }
        }




        void Backward(MD_Vec<Dtype> *bottom,    MD_Vec<Dtype> *top){
            for(int i=0; i<top->count();    i++){
                bottom->diff[i] = top->diff[i];
            }
        }
};


#endif

