#ifndef _SOFTMAX_LOSS_LAYER_HPP_
#define _SOFTMAX_LOSS_LAYER_HPP_


#include"./layer.hpp"
#include"./LayerParameter.hpp"
#include"../utils/utils.hpp"

template <typename Dtype>
class softmax_loss_layer_class// : public layer_class
{
    //private:
    public:

        typedef Dtype value_type;
        int BS;
        int C_in;
        MD_Vec<Dtype>_probs  ;


        softmax_loss_layer_class(const LayerParameter<Dtype>& param, MD_Vec<Dtype> *bottom_0, MD_Vec<Dtype> *bottom_1,    Dtype &top){
            // bottom_0:  logits;    bottom_1: labels
            if(bottom_0->shape.size() == 4){
                bottom_0->shape = {bottom_0->shape[0], bottom_0->count(1,4)};
            }
            CHECK_less(0, bottom_0->shape[0]);
            CHECK_equal(2, int(bottom_0->shape.size()), any2str(__LINE__));
            CHECK_equal(2, int(bottom_1->shape.size()), any2str(__LINE__));
            CHECK_vector_equal(bottom_0->shape,     bottom_1->shape,    "CHECK_vector_equal shape   ");
            BS = bottom_0->shape[0];
            C_in  = bottom_0->shape[1];


            // init top 
            top = Dtype(0);


            // init _labels, _logits,   _probs
            _probs.init_by_shape_and_constant(bottom_0->shape, 0);

        }




        void Forward(MD_Vec<Dtype> *bottom_0,  MD_Vec<Dtype> *bottom_1,  Dtype &top){
            //from logits to probs_
            //  cout<<any2str(__FILE__)+":"+any2str(__LINE__)<<endl;
            //
            compute_softmax(bottom_0, &_probs);
            //from probs_ and labels to loss;
            compute_softmax_cross_entroy_loss(&_probs, bottom_1, top);
        }

        void Backward(MD_Vec<Dtype> *bottom_0,  MD_Vec<Dtype> *bottom_1,  Dtype &top){
            compute_softmax_cross_entroy_loss_backward(bottom_0, &_probs, bottom_1);
        }
};


#endif
