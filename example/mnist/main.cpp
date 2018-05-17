#include"../../src/headers/Layers_headers.hpp"
#include"../../src/utils/utils.hpp"


int main()
{
    // define bottom
    int BS = 100;
    int C_in = 1,   H_in = 28,     W_in = 28;
    int Train_data_number_total = 60000;
    int Test_data_number_total  = 10000;
    int Test_size   = 10000;
    int num_classes = 10;
    int total_steps = 5000000;
    int train_display_step_gap = 500;
    int test_display_step_gap  = 2000;
    double lr = 1e-4;
    int layer_num = 3;
    MD_Vec<double> bottom({BS, C_in, H_in, W_in});
    //  MD_Vec<double> bottom({BS, C_in* H_in* W_in});
    MD_Vec<double> Labels({BS, num_classes});
    vector<MD_Vec<double>> tops(layer_num);
    double loss = 0;
    


    // LayerType,  out_channels,    bias_term_,   lr_w,     lr_b,     
    // filler::w, filler::b, w_std, b_constant,     
    // kernel_size,    stride,     pad
    const LayerParameter<double> conv1_param(LayerType::conv,  20,  true, lr, lr,   filler::gaussian, filler::constant, 0.01, 0.0,  7,  7, 0);
    const LayerParameter<double> fc0_param(LayerType::fc,   num_classes *3,true, lr, lr); 
    const LayerParameter<double> fc1_param(LayerType::fc,   num_classes,true, lr, lr); 
    const LayerParameter<double> softmax_loss_param(LayerType::softmax_loss,  num_classes);


    conv_layer_class<double> Conv1(conv1_param, &bottom, &tops[0]);
    fc_layer_class<double> Fc0(fc0_param, &tops[0], &tops[1]);
    fc_layer_class<double> Fc1(fc1_param, &tops[1], &tops[2]);
    softmax_loss_layer_class<double> Loss(softmax_loss_param, &tops[2],  &Labels, loss);
    
    for(int i=0; i<tops.size(); print_vec(tops[i].shape, "top shape"),  i++);


    vector<double>  X(Train_data_number_total*C_in*H_in*W_in);
    vector<double>  Y(Train_data_number_total*num_classes);
    read_Mnist_Images("../../data/mnist/train-images-idx3-ubyte", X);
    read_Mnist_Labels("../../data/mnist/train-labels-idx1-ubyte", Y);

    vector<double>  X_test(Test_data_number_total*C_in*H_in*W_in);
    vector<double>  Y_test(Test_data_number_total*num_classes);
    read_Mnist_Images("../../data/mnist/t10k-images-idx3-ubyte", X_test);
    read_Mnist_Labels("../../data/mnist/t10k-labels-idx1-ubyte", Y_test);




    long long start_index_X = 0,  end_index_X = 0;
    long long start_index_Y = 0,  end_index_Y = 0;
    for(int step_i=0; step_i<total_steps; step_i++){
        start_index_X = end_index_X >= (long long)(X.size()) ? (end_index_X - (long long)(X.size())) : end_index_X;
        //start_index_X = (step_i * bottom.count()) % (int(X.size())); // will overflow
        
        end_index_X   = start_index_X + bottom.count();

        start_index_Y = end_index_Y >= (long long)(Y.size()) ? (end_index_Y - (long long)(Y.size())) : end_index_Y;
        //start_index_Y = (step_i * Labels.count()) % (int(Y.size())); 
        end_index_Y   = start_index_Y + Labels.count();
        
        //cout<<" step = "<<step_i<<endl;
        //cout<<"start_index_X / X.size(),  start_index_Y / Y.size() = "<<start_index_X<<"/"<<X.size()<<", "<<start_index_Y<<"/"<<Y.size()<<endl;
        copy_vector_from_to<double>(X,  bottom.data, start_index_X, end_index_X);
        copy_vector_from_to<double>(Y,  Labels.data, start_index_Y, end_index_Y);


        // first set_diff_to_zero();
        for(int i=0; i<tops.size(); tops[i].set_diff_to_zero(),  i++);

        int idx = 0;
        Conv1.Forward(&bottom,      &tops[idx]);    idx++;
        Fc0.Forward(&tops[idx-1],   &tops[idx]);    idx++;
        Fc1.Forward(&tops[idx-1],   &tops[idx]);    idx++;
        Loss.Forward(&tops[idx-1],  &Labels, loss); idx++;

        double acc = compute_accuracy(&tops[2], &Labels);
        if(step_i % train_display_step_gap == 0){
            cout<<"--------------------------"<<endl;
            cout<<" train step = "<<step_i<<endl;
            cout<<" train loss = "<<loss<<endl;
            cout<<" train acc  = "<<acc<<endl;
        }


        idx = int(tops.size());
        Loss.Backward(&tops[idx-1], &Labels, loss);     idx --;
        Fc1.Backward(&tops[idx-1],  &tops[idx]);        idx --;
        Fc0.Backward(&tops[idx-1],  &tops[idx]);        idx --;
        Conv1.Backward(&bottom,     &tops[idx]);        idx --;



        if(step_i % test_display_step_gap == 0){
            // ====================================== Test phase ======================================================
            cout<<"testing ....."<<endl;
            int start_index_X_test = 0,     end_index_X_test = 0;
            int start_index_Y_test = 0,     end_index_Y_test = 0;
            double avg_acc  = 0;
            double avg_loss = 0;
            int   test_step = 0;
            CHECK_equal((Test_size * num_classes) % Labels.count(), 0); 
            while(end_index_Y_test + Labels.count() <= Test_size * num_classes){
                //cout<<"end_index_X_test ,end_index_Y_test  = "<<end_index_X_test<<","<<end_index_Y_test<<endl;
                start_index_X_test = end_index_X_test;
                start_index_Y_test = end_index_Y_test;

                end_index_X_test   = start_index_X_test + bottom.count();
                end_index_Y_test   = start_index_Y_test + Labels.count();
                
                copy_vector_from_to<double>(X_test,  bottom.data, start_index_X_test, end_index_X_test);
                copy_vector_from_to<double>(Y_test,  Labels.data, start_index_Y_test, end_index_Y_test);

                int idx = 0;
                Conv1.Forward(&bottom,      &tops[idx]);    idx++;
                Fc0.Forward(&tops[idx-1],   &tops[idx]);    idx++;
                Fc1.Forward(&tops[idx-1],   &tops[idx]);    idx++;
                Loss.Forward(&tops[idx-1],  &Labels, loss); idx++;

                avg_acc  += compute_accuracy(&tops[2], &Labels);
                avg_loss += loss;
                test_step += 1;
                //cout<<"test_step = "<<test_step<<endl;
            }
            cout<<"===> test finished <==="<<endl;
            cout<<endl<<"-->      test_acc  = "<<avg_acc / test_step<<endl;
            cout<<endl<<"-->      test_loss = "<<avg_loss/ test_step<<endl;
            // ====================================== Test phase ======================================================
        }


    }





    return 0;
}

