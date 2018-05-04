#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include"./base_utils.hpp"
#include"./mnist.hpp"


// weight_filler; bias_filler;
enum class filler { constant = 0, gaussian=1, };
enum class LayerType { fc =0, conv=1, relu=2, softmax_loss=101, };




template <typename Dtype>
//void compute_conv2d_by_mat_mul(vector<MD_Vec<Dtype> > &input,   vector<MD_Vec<Dtype>> &output,  MD_Vec<Dtype> &kernel,   int stride,    int pad){
void forward_cpu_bias(MD_Vec<Dtype> *output,  MD_Vec<Dtype> *bias,  MD_Vec<Dtype> *bias_mul){
    CHECK_equal(bias->shape.size(), bias_mul->shape.size());
    CHECK_equal(int(bias->shape.size()),    1);
    CHECK_equal(int(output->count(1)),  int(bias->count() * bias_mul->count()));

    int row_op_A = bias->count(),   col_op_B = bias_mul->count(),  col_op_A = 1; 
    Dtype * p_out_data = &output->data.front(), *p_out_data_end = &output->data.back() + 1;
    while(p_out_data < p_out_data_end){
        mat_mul_use_cblas(&bias->data.front(),      &bias_mul->data.front(),    p_out_data, row_op_A, col_op_B, col_op_A, false, false, Dtype(1));
        p_out_data += output->count(1);
    }
}



template <typename Dtype>
//void compute_conv2d_by_mat_mul(vector<MD_Vec<Dtype> > &input,   vector<MD_Vec<Dtype>> &output,  MD_Vec<Dtype> &kernel,   int stride,    int pad){
void backward_cpu_bias(MD_Vec<Dtype> *output,  MD_Vec<Dtype> *bias,  MD_Vec<Dtype> *bias_mul){
    CHECK_equal(bias->shape.size(), bias_mul->shape.size());
    CHECK_equal(int(bias->shape.size()),    1);
    CHECK_equal(int(output->count(1)),  int(bias->count() * bias_mul->count()));

    int row_op_A = output->count(1,2),   col_op_B = 1,   col_op_A = bias_mul->count();
    Dtype * p_out_diff = &output->diff.front(), *p_out_diff_end = &output->diff.back() + 1;
    while(p_out_diff < p_out_diff_end){
        mat_mul_use_cblas(p_out_diff, &bias_mul->diff.front(), &bias->diff.front(), row_op_A, col_op_B, col_op_A,false,false,Dtype(1));
        p_out_diff += output->count(1);
    }
}





template <typename Dtype>
//void compute_conv2d_by_mat_mul(vector<MD_Vec<Dtype> > &input,   vector<MD_Vec<Dtype>> &output,  MD_Vec<Dtype> &kernel,   int stride,    int pad){
void compute_conv2d_by_mat_mul(MD_Vec<Dtype> *input,   MD_Vec<Dtype> *output,  MD_Vec<Dtype> *kernel,   int stride,    int pad){
    int BS      = input->shape[0];
    int C_in    = input->shape[1],       H_in    = input->shape[2],   W_in    = input->shape[3];
    int C_out   = output->shape[1],      H_out   = output->shape[2],  W_out   = output->shape[3];
    int kernel_size = kernel->shape[3];
    int dim_in  = C_in * H_in * W_in,   dim_out = C_out * H_out * W_out;
	int row_A = C_out;
    int col_A = C_in * kernel_size * kernel_size;
    int row_B = C_in * kernel_size * kernel_size;
    int col_B = H_out * W_out;
    //#pragma omp parallel for
    Dtype *p_in = &input->data[0], *p_out = &output->data[0];
    for(int i=0; i<BS; i++){
        vector<Dtype> input_data_2d(row_B*col_B);
        img2col_by_kernelmat<Dtype>(p_in, &input_data_2d[0], {C_in, H_in, W_in},  {row_B, col_B}, kernel_size, stride, pad);
		// cur_input->shape = {C_in * K_h * K_w,   H_out * W_out}
		// kernel->shape    = {C_out, C_in, K_h, K_w};
		Dtype *A = &kernel->data[0], *B = &input_data_2d[0], *C = p_out;
		mat_mul_use_cblas(A, B, C, row_A, col_B, col_A, false, false);
        p_in += dim_in;     p_out += dim_out;
    }
}





template <typename Dtype>
void compute_conv2d_by_mat_mul_backward(MD_Vec<Dtype> *input,   MD_Vec<Dtype> *output,  MD_Vec<Dtype> *kernel,   int stride,    int pad){
    int BS      = input->shape[0];
    int C_in    = input->shape[1],       H_in    = input->shape[2],   W_in    = input->shape[3];
    int C_out   = output->shape[1],      H_out   = output->shape[2],  W_out   = output->shape[3];
    int kernel_size = kernel->shape[3];
    int dim_in  = C_in * H_in * W_in,   dim_out = C_out * H_out * W_out;

    int row_kernel = C_out;
    int col_kernel = C_in * kernel_size * kernel_size;
    int row_input  = C_in * kernel_size * kernel_size;
    int col_input  = H_out * W_out;
    int row_output = C_out;
    int col_output = H_out * W_out;
    CHECK_equal(C_out,  kernel->shape[0]);
    CHECK_equal(C_in,   kernel->shape[1]);
    CHECK_equal(H_out,  (H_in - kernel_size + 2 * pad) / stride + 1 );
    CHECK_equal(W_out,  (W_in - kernel_size + 2 * pad) / stride + 1 );
    //#pragma omp parallel for
    Dtype *p_in_data = &input->data[0], *p_out_data = &output->data[0];
    Dtype *p_in_diff = &input->diff[0], *p_out_diff = &output->diff[0];
    for(int i=0; i<BS; i++){
        vector<Dtype> input_diff_2d(row_input* col_input);
        vector<Dtype> input_data_2d(row_input* col_input);

        // get col input_data_2d
        img2col_by_kernelmat<Dtype>(p_in_data, &input_data_2d[0], {C_in, H_in, W_in},  {row_input, col_input}, kernel_size, stride, pad);
		// cur_input.shape = {C_in * K_h * K_w,   H_out * W_out}
		// kernel.shape    = {C_out, C_in, K_h, K_w};
        // output.shape    = {C_out, H_out*W_out};
		//void  mat_mul_use_cblas(A, B, C, row_A, col_B, col_A, false,	true);

        // from output diff to cur_input diff
		mat_mul_use_cblas(&kernel->data[0], p_out_diff, &input_diff_2d[0], col_kernel, col_output, row_kernel,  true, false);
        // col2img_by_kernelmat  contains add operation
        col2img_by_kernelmat<Dtype>(p_in_diff, &input_diff_2d[0], {C_in, H_in, W_in}, {row_input, col_input},  kernel_size,  stride,  pad);

        // from output diff to kernel diff,  diff = old_diff + new_diff,  so  the final need 1.0 
		mat_mul_use_cblas(p_out_diff, &input_data_2d[0], &kernel->diff[0], row_output, row_input, col_output, false, true, 1.0);
        p_in_data += dim_in;    p_out_data += dim_out;
        p_in_diff += dim_in;    p_out_diff += dim_out;
    }
}










template <typename Dtype>
void compute_conv2d_by_7_for(MD_Vec<Dtype> *input,   MD_Vec<Dtype> *output,  MD_Vec<Dtype> *kernel,   int stride,    int pad=0){

    int BS      = input->shape[0];
    int C_in    = input->shape[1],       H_in    = input->shape[2],   W_in    = input->shape[3];
    int C_out   = output->shape[1],      H_out   = output->shape[2],  W_out   = output->shape[3];
    int kernel_size = kernel->shape[3];
    int dim_in  = C_in * H_in * W_in,   dim_out = C_out * H_out * W_out;

	int row_A = C_out;
    int col_A = C_in * kernel_size * kernel_size;
    int row_B = C_in * kernel_size * kernel_size;
    int col_B = H_out * W_out;

    CHECK_equal(C_out,  kernel->shape[0]);
    CHECK_equal(C_in,   kernel->shape[1]);
    CHECK_equal(H_out,  (H_in - kernel_size + 2 * pad) / stride + 1 );
    CHECK_equal(W_out,  (W_in - kernel_size + 2 * pad) / stride + 1 );
    //#pragma omp parallel for
    Dtype *p_in_data  = &input->data[0];
    Dtype *p_out_data = &output->data[0];
    Dtype *top, *bot, *ker;
    for(int bs = 0; bs < BS;    p_out_data += dim_out,   p_in_data += dim_in,    bs++){
        top = p_out_data;
        ker = &kernel->data[0];
        for(int c_out=0; c_out < C_out; c_out++,  top += H_out * W_out){
            bot = p_in_data;
            for(int c_in=0; c_in < C_in; c_in ++, bot += H_in * W_in,   ker += kernel_size * kernel_size){
                for(int h_out=0; h_out < H_out; h_out ++){
                    for(int w_out=0; w_out < W_out; w_out ++){
                        for(int kh=0; kh < kernel_size; kh ++){
                            for(int kw=0; kw < kernel_size; kw ++){
                                int h_in = h_out*stride + kh ,   w_in = w_out*stride + kw;
                                //  CHECK_less(h_in, H_in, any2str(__LINE__));
                                //  CHECK_less(w_in, W_in, any2str(__LINE__));
                                //  CHECK_less(bot + h_in * W_in + w_in, &input[bs].data.back() + 1, any2str(__LINE__));
                                //  CHECK_less(ker + kh * kernel_size + kw, &kernel.data.back() + 1, any2str(__LINE__));
                                //  CHECK_less(top + h_out * W_out + w_out,  &output[bs].data.back() + 1,any2str(__LINE__));
                                *(top + h_out * W_out + w_out) += (*(bot + h_in * W_in + w_in)) * (*(ker + kh * kernel_size + kw));
                            }
                        }
                    }
                }
            }
        }
    }
}








// ===============================   relu   ===================================

template <typename Dtype>
void math_RELU(MD_Vec<Dtype> *input){
    int total_size = input->data.size();
    Dtype *p = &input->data[0];
    for(int i=0; i<total_size; i++){
        *p = (*p) > 0 ? (*p) : 0;
        p ++;
    }
}



template <typename Dtype>
void math_RELU_backward(MD_Vec<Dtype> *input){
    int total_size = input->data.size();
    Dtype *pdata = &input->data[0];
    Dtype *pdiff = &input->diff[0];
    for(int i=0; i<total_size; i++){
        *pdiff = (*pdata) > 0 ? (*pdiff) : 0;
        pdiff ++;   pdata ++;
    }
}



// ===============================   relu   ===================================






// ===============================   fc   ===================================

template <typename Dtype>
void compute_fc_by_mat_mul(MD_Vec<Dtype> *input,   MD_Vec<Dtype> *output,  MD_Vec<Dtype> *kernel){
    int BS = input->shape[0];
    int C_out   = output->count(1);
    int C_in    = input->count(1);
    CHECK_equal(C_out, kernel->shape[0], any2str(__LINE__));
    CHECK_equal(C_in,  kernel->shape[1], any2str(__LINE__));
	int row_A = C_out;
    int col_A = C_in;
    int row_B = C_in;
    int col_B = 1;
    Dtype *p_in_data = &input->data[0];
    Dtype *p_out_data= &output->data[0];
    for(int i=0; i<BS;  p_in_data += C_in,   p_out_data += C_out,   i++){
		Dtype *A = &kernel->data[0], *B = p_in_data, *C = p_out_data;
        //cout<<" start mat_mul "<<endl;
		mat_mul_use_cblas(A, B, C, row_A, col_B, col_A, false, false);
        //mat_mul_self_implement_2(A, B, C, row_A, col_B, col_A);
    }
}





template <typename Dtype>
void compute_fc_by_mat_mul_backward(MD_Vec<Dtype> *input,   MD_Vec<Dtype> *output,  MD_Vec<Dtype> *kernel){
    int BS = input->shape[0];
    int C_out   = output->count(1);
    int C_in    = input->count(1);
    CHECK_equal(C_out, kernel->shape[0], any2str(__FILE__)+ ":" +any2str(__LINE__));
    CHECK_equal(C_in,  kernel->shape[1], any2str(__FILE__)+ ":" +any2str(__LINE__));

    Dtype *p_in_data = &input->data[0];     
    Dtype *p_in_diff = &input->diff[0],     *p_out_diff= &output->diff[0];

    for(int i=0; i<BS; p_in_diff += C_in,   p_in_data += C_in,  p_out_diff += C_out,    i++){
		//Dtype *A = &kernel.data[0], *B = &input[i].data[0], *C = &output[i].data[0];
		//mat_mul_use_cblas(A, B, C, row_A, col_B, col_A, false, false);
        //
        // from output[i].diff  to input[i].diff
        // [1, C_out]  * [C_out, C_in];
		Dtype *A = p_out_diff, *B = &kernel->data[0], *C = p_in_diff;
		mat_mul_use_cblas(A, B, C, 1, C_in, C_out, false, false, Dtype(1.0));

        // from output[i].diff  to kernel[i].diff
        // [C_out, 1] * [1,  C_in]
		A = p_out_diff, B = p_in_data, C = &kernel->diff[0];
		mat_mul_use_cblas(A, B, C, C_out, C_in, 1, false, false, Dtype(1.0));
    }
}

// ===============================   fc   ===================================



// ===============================  softmax =================================
template <typename Dtype>
void compute_softmax(MD_Vec<Dtype> *input,   MD_Vec<Dtype> *output){
    // use a trick from caffe;      substract the max_val of cur_row
    int BS      = input->shape[0];
    int C_in_out= input->shape[1];
    CHECK_less(0, BS);
    CHECK_equal(int(input->shape.size()),   2);
    CHECK_equal(int(output->shape.size()),  2);
    Dtype *p_in_data = &input->data[0], *p_out_data = &output->data[0];
    for(int i =0; i<BS; p_in_data += C_in_out, p_out_data += C_in_out, i++){
        Dtype row_sum = 0;
        Dtype max_val = INT_MIN;
        for(int j=0; j<C_in_out; max_val = *(p_in_data + j) > max_val ? *(p_in_data + j) : max_val,     j++);
        for(int j=0; j<C_in_out; *(p_out_data + j) = *(p_in_data + j) - max_val,                        j++);
        for(int j=0; j<C_in_out; row_sum += std::exp(*(p_out_data + j)),                                j++);
        for(int j=0; j<C_in_out;                                                                        j++){
            *(p_out_data + j) = std::exp(*(p_out_data + j)) / row_sum;
        }
    }
}

// ===============================  softmax =================================







// ===============================  softmax_cross_entroy_loss =================================
template <typename Dtype>
void compute_softmax_cross_entroy_loss(MD_Vec<Dtype> *probs, MD_Vec<Dtype> *labels,  Dtype &loss){
    // loss = mean({log(p_i)})
    int BS = probs->shape[0];
    int C_in_out = probs->shape[1];
    int total_size = BS * C_in_out;
    CHECK_equal(BS, int(labels->shape[0]));    CHECK_less(0, BS);
    Dtype *p_probs_data = &probs->data[0],     *p_labels_data = &labels->data[0];
    loss = 0;
    for(int i=0; i<total_size;  p_probs_data ++, p_labels_data ++,  i++){
        //  if(i % C_in_out == 0) cout<<"------------------------"<<endl;
        //  cout<<"*p_probs_data = "<<*p_probs_data<<endl;
        if(*p_labels_data == 0) continue;
        loss += - std::log(std::max(1e-10, *(p_probs_data)));
    }
    loss = loss / BS;
}



template <typename Dtype>
void compute_softmax_cross_entroy_loss_backward(MD_Vec<Dtype> *logits,  MD_Vec<Dtype> *probs,   MD_Vec<Dtype> *labels){
    // just need backward to logits_diff
    int BS = probs->shape[0];
    int C_in_out = probs->shape[1];
    int total_size = BS * C_in_out;
    CHECK_equal(BS, int(labels->shape[0]));    CHECK_less(0, BS);
    Dtype *p_probs_data = &probs->data[0],     *p_labels_data = &labels->data[0];
    Dtype *p_logits_diff = &logits->diff[0];

    for(int i=0; i<total_size; i++, p_logits_diff ++,   p_probs_data ++,    p_labels_data ++){
        *p_logits_diff = (1.0 / BS) * (*p_probs_data - *p_labels_data);
    }
}
// ===============================  softmax_cross_entroy_loss =================================


// ===============================  dw_conv =================================
// ===============================  dw_conv =================================




// ===============================  accuracy =================================
template <typename Dtype>
Dtype compute_accuracy(MD_Vec<Dtype> *logits,  MD_Vec<Dtype> *labels){
    CHECK_vector_equal(logits->shape, labels->shape, "logits.shape and labels.shape equal ");
    int BS = logits->shape[0],   class_num = logits->shape[1];
    Dtype *p_logits_data = &logits->data.front();
    Dtype *p_labels_data = &labels->data.front();
    int correct_cnt = 0;
    for(int i=0; i<BS; i++, p_logits_data += class_num, p_labels_data += class_num){
        int idx_label = 0;
        for(int j=0; j<class_num; j++){
            if(p_labels_data[j] == 1){
                idx_label = j;  break;
            }
        }
        int idx_logits =0;
        for(int j=0; j<class_num; j++){
            idx_logits = p_logits_data[j] > p_logits_data[idx_logits] ? j : idx_logits;
        }
        if(idx_label == idx_logits) correct_cnt += 1;
    }
    //cout<<"correct_cnt = "<<correct_cnt<<endl;
    return Dtype(correct_cnt) / BS;
}
// ===============================  accuracy =================================




// ===============================  drop out =================================
template <typename Dtype>
void compute_dropout(MD_Vec<Dtype> *bottom, MD_Vec<Dtype> *top, Dtype dropout_ratio, Dtype* keep_mask, bool is_training){
    //  get_bernoulli_vec<Dtype>(bottom->count(), 1 - dropout_ratio,  keep_mask);
    //  ensure the engine not loss,  need to multiply (1 / (1 - dropout_ratio));
    if(is_training){
        Dtype scale_ = (1 / (1 - dropout_ratio));
        for(int i=0; i<bottom->count(); i++){
            top->data[i] = bottom->data[i] * (keep_mask[i] * scale_);
        }
    }
    else{
        copy_vector_from_to(bottom->data, top->data, 0, bottom->count());
    }
}


template <typename Dtype>
void compute_dropout_backward(MD_Vec<Dtype> *bottom, MD_Vec<Dtype> *top, Dtype dropout_ratio, Dtype* keep_mask, bool is_training){
    if(is_training){
        Dtype scale_ = (1 / (1 - dropout_ratio));
        for(int i=0; i<bottom->count(); i++){
            bottom->diff[i] = top->diff[i] * (keep_mask[i] * scale_);
        }
    }
    else{
        copy_vector_from_to(top->diff, bottom->diff, 0, bottom->count());
    }
}
// ===============================  drop out =================================





#endif

