#include"../../src/utils//utils.hpp"

void test_CHECK_equal(){
    CHECK_equal_by_diff<double>(1e-6, 1.5e-6, " diff = 1e-6 ", 1e-6);
    CHECK_equal_by_diff<double>(1e-7, 1.2e-7, " diff = 1e-7 ", 1e-7);
    CHECK_vector_equal_test_success_by_diff(vector<double>{1.1e-4, 1.2e-4}, vector<double>{1.1001e-4, 1.200001e-4}, __FUNCTION__);
}


void test_get_type_str(){
    int i =9;
    int *pi = &i;
    float f = 9;
    double d = 9;
    string st_i = "i";
    string st_pi = "Pi";
    string st_f = "f";
    string st_d = "d";
    CHECK_equal(get_type_str(i),  st_i);
    CHECK_equal(get_type_str(pi), st_pi);
    CHECK_equal(get_type_str(f),  st_f);
    CHECK_equal(get_type_str(d),  st_d);
    cout<<" "<<__FUNCTION__<<" success finished !! "<<endl;
}


void get_conv_test_data(MD_Vec<double> &bs_in, MD_Vec<double> &bs_out, MD_Vec<double>& kernel){
    int C_in = 2,  H_in = 3, W_in = 3, stride = 1, kernel_size = 2, C_out = 3, pad = 0, bs = 1;
    int H_out = (H_in - kernel_size + 2*pad) / stride + 1;
    int W_out = (W_in - kernel_size + 2*pad) / stride + 1;
    //build  input
    bs_in.data = {1,1,1,2,2,2,3,3,3,  10,10,10,20,20,20,30,30,30};
    bs_in.diff = {1,0,0,0,0,0,0,0,-1,  1,  1,2,  0,0,0, 0,0,0};
    bs_in.shape= {bs, C_in, H_in, W_in};
    //build kernel 
    kernel.data= {1,1,1,1, 2,2,2,2,  1,1,1,1,  2,2,2,2,  1,1,1,1,  2,2,2,2};
    kernel.diff= {0,0,0,0, -1,2,1,0, 0,0,0,0,  1,1,1,0,  0,0,0,0,  0,0,1,0};
    kernel.shape={C_out, C_in, kernel_size, kernel_size};
    //init output
    bs_out.data.resize(C_out*H_out*W_out);
    bs_out.diff = {1,0,1,2,  1,0,0,1,  0,3,0,1};
    bs_out.shape= {bs,  C_out, H_out, W_out};
}


void get_softmax_test_data(MD_Vec<double> &logits, MD_Vec<double> &probs, MD_Vec<double> &labels){
    int C_in_out = 3,   bs = 2;
    vector<double> logits_data = {1, 3, 1,  1,  2, 2};
    vector<double> logits_diff = {1, 1, 0,  -1, 0, 2};
    vector<double> probs_data = {1, 0, 0,  0,  1, 0};
    vector<double> probs_diff = {0, 2, 1,  2,-1, 0};
    //build  input
    logits.data = logits_data;
    logits.diff = logits_diff; 
    logits.shape= {bs,  C_in_out};
    //init output
    probs.data  = probs_data;
    probs.diff  = probs_diff;
    probs.shape = {bs,   C_in_out};


    labels.data = probs_data;
    labels.diff = probs_diff;
    labels.shape= {bs,  C_in_out};
}




void test_mat_mul_run_time(){
    int row_A   =  300;
    int col_B   =  500;
    int col_A   =  300;
    int size_A  =  row_A*col_A;
    int size_B  =  col_A*col_B;
    int size_C  =  row_A*col_B;
    cout<<"row_A,col_B,col_A = "<<row_A<<","<<col_B<<","<<col_A<<endl;
    //double A[row_A*col_A];
    //double B[col_A*col_B];
    //double C[row_A*col_B];
    //static double A[size_A];
    //static double B[size_B];
    //static double C[size_C];
    vector<double> vec_A(size_A);
    vector<double> vec_B(size_B);
    vector<double> vec_C(size_C);

    double * A = &vec_A[0];
    double * B = &vec_B[0];
    double * C = &vec_C[0];


    clock_t start = clock();
    get_rand_c_array<double>(A, row_A*col_A);
    get_rand_c_array<double>(B, col_A*col_B);
    //print_c_array(C, row_A*col_B);
    //fill_n(C, row_A*col_B, 0);
    //print_c_array(C, row_A*col_B);
    //cout<<"------start mat_mul"<<endl;
    start = clock();
    mat_mul_use_cblas<double> (A, B, C, row_A, col_B, col_A);
    print_time_using(start, "mat_mul_use_cblas using time = ");
    if(size_A < 50) print_c_array(C, row_A*col_B);
    //print_c_array(C, row_A*col_B);


    start = clock();
    mat_mul_self_implement_1<double>(A, B, C, row_A, col_B, col_A);
    print_time_using(start, "mat_mul_self_implement_1 using time = ");
    if(size_A < 50) print_c_array(C, row_A*col_B);
    //print_c_array(C, row_A*col_B);


    start = clock();
    mat_mul_self_implement_2<double>(A, B, C, row_A, col_B, col_A);
    print_time_using(start, "mat_mul_self_implement_2 using time = ");
    if(size_A < 50) print_c_array(C, row_A*col_B);
    //print_c_array(C, row_A*col_B);

}




void test_mat_mul(){
    int row_A   =  3;
    int col_B   =  3;
    int col_A   =  2;
    int size_A  =  row_A*col_A;
    int size_B  =  col_A*col_B;
    int size_C  =  row_A*col_B;

    vector<double> vec_A = {1,2,3,4,5,6};
    vector<double> vec_B = {0,1,2,3,4,5};
    vector<double> vec_C = {0,0,0,0,0,0,0,0,0};
    vector<double> expect_vec = {6,9,12,12,19,26,18,29,40};
    double * A = &vec_A[0],  * B = &vec_B[0],    * C = &vec_C[0];
    
    vec_C = {0,0,0,0,0,0,0,0,0};
    mat_mul_use_cblas<double> (A, B, C, row_A, col_B, col_A);
    CHECK_vector_equal_test_success(vec_C, expect_vec, "mat_mul_use_cblas");

    vec_C = {0,0,0,0,0,0,0,0,0};
    mat_mul_self_implement_1<double> (A, B, C, row_A, col_B, col_A);
    CHECK_vector_equal_test_success(vec_C, expect_vec, "mat_mul_self_implement_1");


    vec_C = {0,0,0,0,0,0,0,0,0};
    mat_mul_self_implement_2<double> (A, B, C, row_A, col_B, col_A);
    CHECK_vector_equal_test_success(vec_C, expect_vec, "mat_mul_self_implement_2");



    expect_vec = {2,8,14,4,18,32,6,28,50};
    vec_C = {0,0,0,0,0,0,0,0,0};
    mat_mul_use_cblas<double> (A, B, C, row_A, col_B, col_A,false,true);
    print_vec(vec_C);
    CHECK_vector_equal_test_success(vec_C, expect_vec, "mat_mul_use_cblas");



    expect_vec = {3,8,16,4,21,32,5,28,50};
    //vec_C = {0,0,0,0,0,0,0,0,0};
    vec_C = {1,0,2,0,3,0,-1,0,0};
    mat_mul_use_cblas<double> (A, B, C, row_A, col_B, col_A,false,true, 1.0);
    print_vec(vec_C);
    CHECK_vector_equal_test_success(vec_C, expect_vec, "mat_mul_use_cblas");

}





void test_vec_address(){
    int vec_1d[10]; get_rand_c_array(vec_1d, 10);
    print_c_array(vec_1d, 10);
    int *p = vec_1d;
    //  int address_int = &vec_1d[0];
    //  invalid conversion from ‘int*’ to ‘int’
    for(int i=0; i<10; i++){
        //cout<<"&vec_1d[i], vec_1d[i], p, *p = "<<&vec_1d[i]<<","<<vec_1d[i]<<";  "<<p<<","<<*p<<endl;
        CHECK_equal(&vec_1d[i], p, any2str(__FILE__) +  " :line: " + any2str(__LINE__));
        CHECK_equal(vec_1d[i], *p, any2str(__FILE__) +  " :line: " + any2str(__LINE__));
        CHECK_equal((&vec_1d[i])+1, p+1, any2str(__FILE__) +  " :line: " + any2str(__LINE__));
        //CHECK_equal(address_int, p, any2str(__FILE__) +  " :line: " + any2str(__LINE__));
        //address_int += 4;
        p++;
    }
    cout<<" function : "<<__FUNCTION__<<" success finished !!! "<<endl;
}




void test_load_txt_to_vector_string(){
    const char* txt = "./test_txt.txt";
    vector<string> vec_str;
    load_txt_to_vector_string(txt, vec_str);
    vector<string> expect_vec =  {"abcd efg", "--------------",  "0000000",  "12345"};

    CHECK_vector_equal_test_success(vec_str, expect_vec, __FUNCTION__);
}



void test_any2str(){
    int num_int = 100;
    string expect_str = "100";
    string s = any2str<int>(num_int);
    CHECK_equal(s, expect_str);


    float num_float = 10.98;
    expect_str = "10.98";
    s = any2str<float>(num_float);
    CHECK_equal(s, expect_str);
}



void test_split_string(){
    string st = "abc#hello#ui#hello#mkl#00#helll#1";
    //string seq = "hello";
    string seq = "hello";
    vector<string> res = split_string(st, seq);
    vector<string> expect_vec = {"abc#","#ui#", "#mkl#00#helll#1"};
    CHECK_vector_equal_test_success(expect_vec, res, __FUNCTION__);
}



void test_img2col_by_kernelmat(){
    int C_in = 2,  H_in = 3, W_in = 3, stride = 1, kernel_size = 2;
    int H_out = (H_in - kernel_size) / stride + 1,      W_out = H_out;
    vector<int> vec3d = {1,1,1,2,2,2,3,3,3,10,10,10,20,20,20,30,30,30};//get_rand_vec(C_in * H_in * W_in);
    vector<int> vec2d(C_in * kernel_size * kernel_size * H_out * W_out, 0);
    img2col_by_kernelmat<int>(&vec3d[0], &vec2d[0], {C_in, H_in, W_in}, {C_in*kernel_size*kernel_size,  H_out*W_out}, kernel_size, stride);
    vector<int> expect_vec = {1,1,2,2,1,1,2,2,2,2,3,3,2,2,3,3,10,10,20,20,10,10,20,20,20,20,30,30,20,20,30,30};
    CHECK_vector_equal_test_success(expect_vec, vec2d, __FUNCTION__);
}






void test_col2img_by_kernelmat(){
    int C_in = 2,  H_in = 3, W_in = 3, stride = 1, kernel_size = 2;
    int H_out = (H_in - kernel_size) / stride + 1,      W_out = H_out;
    vector<int> vec3d = {1,1,1,2,2,2,3,3,3,10,10,10,20,20,20,30,30,30};
    //get_rand_vec(C_in * H_in * W_in);
    vector<int> vec2d(C_in * kernel_size * kernel_size * H_out * W_out, 0);
    img2col_by_kernelmat<int>(&vec3d[0], &vec2d[0], {C_in, H_in, W_in}, {C_in*kernel_size*kernel_size,  H_out*W_out}, kernel_size, stride);
    vector<int> expect_vec = {1,1,2,2,1,1,2,2,2,2,3,3,2,2,3,3,10,10,20,20,10,10,20,20,20,20,30,30,20,20,30,30};
    CHECK_vector_equal(expect_vec, vec2d);


    vector<int> new_vec3d(C_in* H_in *W_in);
    col2img_by_kernelmat(&new_vec3d[0], &vec2d[0], {C_in, H_in, W_in}, {C_in*kernel_size*kernel_size,  H_out*W_out}, kernel_size, stride);
    


    expect_vec = {1,2,1,4,8,4,3,6,3,10,20,10,40,80,40,30,60,30};
    print_vec(new_vec3d, "new_vec3d =");
    CHECK_vector_equal_test_success(expect_vec, new_vec3d, __FUNCTION__);

}







void test_compute_conv2d_by_mat_mul(){
    int C_in = 2,  H_in = 3, W_in = 3, stride = 1, kernel_size = 2, C_out = 3, pad = 0, bs = 1;
    int H_out = (H_in - kernel_size) / stride + 1;
    int W_out = (W_in - kernel_size) / stride + 1;
    MD_Vec<double> bs_in({bs, C_in, H_in, W_in});
    MD_Vec<double> kernel({C_out, C_in, kernel_size, kernel_size});
    MD_Vec<double> bs_out({bs, C_out, H_out, W_out});
    get_conv_test_data(bs_in, bs_out, kernel);
    // expect_vec
    vector<double> expect_vec = { 126,126,210,210,126,126,210,210,126,126,210,210 };

    // start conv2d 
    compute_conv2d_by_mat_mul(&bs_in,  &bs_out, &kernel, stride, pad);

    CHECK_vector_equal_test_success(expect_vec, bs_out.data, __FUNCTION__);
}



void test_transpose_matrix(){

    vector<double> nums = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    vector<double> expect_vec = {1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16};
    //print_vec(nums, "shape = {8,2}");
    transpose_matrix(nums, {8,2});

    CHECK_vector_equal_test_success(expect_vec, nums, __FUNCTION__);

}






void test_compute_conv2d_by_7_for(){
    int C_in = 2,  H_in = 3, W_in = 3, stride = 1, kernel_size = 2, C_out = 3, pad = 0, bs = 1;
    int H_out = (H_in - kernel_size + 2*pad) / stride + 1;
    int W_out = (W_in - kernel_size + 2*pad) / stride + 1;
    MD_Vec<double> bs_in({bs, C_in, H_in, W_in});
    MD_Vec<double> bs_out({bs,C_out,H_out,W_out});
    MD_Vec<double> kernel({C_out, C_in, kernel_size, kernel_size});
    get_conv_test_data(bs_in, bs_out, kernel);
    // expect_vec
    vector<double> expect_vec = { 126,126,210,210,126,126,210,210,126,126,210,210 };
    // start conv2d 
    compute_conv2d_by_7_for(&bs_in,  &bs_out, &kernel, stride, pad);
    //CHECK_vector_equal(expect_vec, bs_out[0].data);
    //cout <<"    "<<__FUNCTION__<<"    success finished "<<endl;
    CHECK_vector_equal_test_success(expect_vec, bs_out.data, __FUNCTION__);
}




void test_compute_conv2d_by_mat_mul_backward(){
    int C_in = 2,  H_in = 3, W_in = 3, stride = 1, kernel_size = 2, C_out = 3, pad = 0, bs = 1;
    int H_out = (H_in - kernel_size) / stride + 1;
    int W_out = (W_in - kernel_size) / stride + 1;
    //build  input
    MD_Vec<double> bs_in({bs, C_in, H_in, W_in});
    MD_Vec<double> kernel({C_out, C_in, kernel_size, kernel_size});
    MD_Vec<double> bs_out({bs,C_out, H_out, W_out});
    get_conv_test_data(bs_in, bs_out, kernel);
    // expect_vec
    vector<double> expect_vec_out_data = {126,126,210,210,126,126,210,210,126,126,210,210 };
    vector<double> expect_vec_out_diff = {1,0,1,2,  1,0,0,1,  0,3,0,1};
    //vector<double> expect_vec_inp_diff = {2,5,3,  3,10,7,  1,5,4, 4,10,6, 6,20,14, 2,10,8};
    vector<double> expect_vec_inp_diff = {3,5,3,  3,10,7,  1,5,3, 5,11,8, 6,20,14, 2,10,8};
    vector<double> expect_vec_ker_diff = {7,7,11,11,69,72,111,110, 3,3,5,5,31,31,51,50, 5,5,9,9,50,50,91,90};


    // start conv2d 

    compute_conv2d_by_mat_mul(&bs_in,  &bs_out, &kernel, stride, pad);

    compute_conv2d_by_mat_mul_backward(&bs_in,  &bs_out, &kernel, stride, pad);


    CHECK_vector_equal_test_success(expect_vec_out_data, bs_out.data, any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success(expect_vec_out_diff, bs_out.diff, any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success(expect_vec_inp_diff, bs_in.diff, any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success(expect_vec_ker_diff, kernel.diff, __FUNCTION__);
}





void test_math_RELU_forward_and_backward(){
    MD_Vec<double> input({3, 2});
    input.data = {1,2,3,-4, -1, 0};
    input.diff = {0,1,0, 2, -1, 0};
    input.shape= {3,2};

    vector<double> expect_vec_inp_data = {1,2,3,0,0,0};
    vector<double> expect_vec_inp_diff = {0,1,0,0,0,0};
    vector<int> expect_vec_inp_shap = {3,2};
    
    math_RELU(&input);
    math_RELU_backward(&input);


    CHECK_vector_equal_test_success(expect_vec_inp_data, input.data, any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success(expect_vec_inp_diff, input.diff, any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success(expect_vec_inp_shap, input.shape, __FUNCTION__);
}








void test_compute_fc_by_mat_mul_forward_and_backward(){
    int BS = 1,  C_in = 4,   C_out = 2;
    cout<<any2str(__FILE__) +  " :line: " + any2str(__LINE__)<<endl;
    MD_Vec<double> bs_in({BS, C_in});
    bs_in.data = {1, 0,2, 1};
    bs_in.diff = {1,-1,0, 2};
    bs_in.shape= {BS,  C_in};

    cout<<any2str(__FILE__) +  " :line: " + any2str(__LINE__)<<endl;
    MD_Vec<double> kernel;
    kernel.data = {1,2,1,0,  2,0, 1,1};
    kernel.diff = {1,0,1,0,  0,0,-1,1};
    kernel.shape= {C_out, C_in};


    MD_Vec<double> bs_out({BS,C_out});
    bs_out.data = {0, 2};
    bs_out.diff = {-1,9};
    bs_out.shape= {BS,  C_out};
    // expect_vec
    vector<double> expect_bs_in_data  = bs_in.data;
    vector<double> expect_bs_in_diff  = {18,-3,8,11};
    vector<double> expect_bs_out_data = {3, 5};
    vector<double> expect_bs_out_diff = bs_out.diff;
    vector<double> expect_kernel_data = kernel.data;
    vector<double> expect_kernel_diff = {0,0,-1,-1,9,0,17,10};

    cout<<any2str(__FILE__) +  " :line: " + any2str(__LINE__)<<endl;
    // compute_fc_by_mat_mul
    compute_fc_by_mat_mul(&bs_in, &bs_out, &kernel);
    compute_fc_by_mat_mul_backward(&bs_in, &bs_out, &kernel);
   
    CHECK_vector_equal_test_success(expect_bs_in_data,  bs_in.data,  any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success(expect_bs_in_diff,  bs_in.diff,  any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success(expect_bs_out_data, bs_out.data, any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success(expect_bs_out_diff, bs_out.diff, any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success(expect_kernel_data, kernel.data,    any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success(expect_kernel_diff, kernel.diff,    __FUNCTION__);
}








void test_compute_softmax_and_softmax_loss_forward_backward(){
    int C_in_out = 3, BS = 2;
    MD_Vec<double> logits({BS,  C_in_out});
    MD_Vec<double> probs({BS,   C_in_out});
    MD_Vec<double> labels({BS,  C_in_out});
    get_softmax_test_data(logits, probs, labels);
    double loss = 18;

    vector<double> expect_probs_data  = {0.10650698,  0.78698611,  0.10650698,      0.15536238,  0.42231879,  0.42231879};
    vector<double> expect_probs_diff = probs.diff;
    vector<double> expect_logits_data= logits.data;
    vector<double> expect_logits_diff= {-0.44674651,  0.39349306,  0.05325349,     0.07768119, -0.28884061,  0.21115939};
    vector<double> expect_labels_data = labels.data;
    vector<double> expect_labels_diff = labels.diff;
    double expect_loss = 1.5507697898355257;

    compute_softmax(&logits, &probs);
    compute_softmax_cross_entroy_loss(&probs, &labels, loss);
    compute_softmax_cross_entroy_loss_backward(&logits, &probs, &labels);

    CHECK_vector_equal_test_success_by_diff(expect_logits_data,  logits.data,  any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success_by_diff(expect_logits_diff,  logits.diff,  any2str(__FILE__) +  " :line: " + any2str(__LINE__));

    CHECK_vector_equal_test_success_by_diff(expect_probs_data,  probs.data,  any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success_by_diff(expect_probs_diff,  probs.diff,  any2str(__FILE__) +  " :line: " + any2str(__LINE__));

    CHECK_vector_equal_test_success_by_diff(expect_labels_data,  labels.data,  any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success_by_diff(expect_labels_diff,  labels.diff,  any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success_by_diff(vector<double>{expect_loss}, vector<double>{loss}, __FUNCTION__);

}


void test_init_by_MDVec(){
    MD_Vec<double> A({3,3}, 11);
    MD_Vec<double> B({3,3});
    //  clock_t start = clock();
    //  A.copy_to(B);
    //  print_time_using(start, "===> function: test_init_by_MDVec  copy_to  using time = ");
    //  MD_Vec<double> B;
    B.init_by_MDVec(A);
    //  A.print("A = ");
    //  B.print("B = ");
    A.CHECK_EQ(B);
    CHECK_vector_equal_test_success(A.data, B.data, __FUNCTION__);
}



void test_get_random_Labels(){
    int BS = 3, num_classes = 3;
    vector<double> labels = get_random_Labels<double>({BS, num_classes});
    //print_vec(labels, " labels = ");
}


void test_copy_vector_from_to(){
    vector<int> A = {1,2,3,12,13,14};
    vector<int> B = {210,111,121,112,231,113};
    //  copy_vector_from_to(A, B, 0, 6);
    //  CHECK_vector_equal_test_success(A, B, __FUNCTION__);

    copy_vector_from_to(A, B, 3, 7);
    vector<int> expect_B = {12,13,14,1,231,113};
    CHECK_vector_equal_test_success(expect_B, B, __FUNCTION__);

}




void test_get_rand_vec(){
    vector<double> A = get_rand_vec<double>(10, 4, 0.1);
    print_vec(A, "(get_rand_vec()  mean =  4,  std = 0.1) = ");
}



void test_get_random_data_and_label(){
    int data_num = 12;   int feature_num = 5;    int class_num = 3;
    vector<double> X(data_num * feature_num, -1);
    vector<double> Y(data_num * class_num, -1);
    get_random_data_and_label<double>(X, Y, data_num, feature_num, class_num);
    print_vec<double>(X, "X = ");
    print_vec<double>(Y, "Y = ");
}



void test_compute_accuracy(){
    int data_num = 5;   int feature_num = 2;    int class_num = 2;
    vector<double> X = {1.1,0.9, 0.2,1.8, 1.6,0.4, 0.9,0.2, 1.1,1.0};
    vector<double> Y = {0,1,    1,0,      1,0,      1,0,     1,0};
    print_vec<double>(X, "X = ");
    print_vec<double>(Y, "Y = ");
    MD_Vec<double> X_({data_num, feature_num});
    MD_Vec<double> Y_({data_num, class_num});
    copy_vector_from_to(X, X_.data, 0, X_.count());
    copy_vector_from_to(Y, Y_.data, 0, X_.count());
    double acc = compute_accuracy(&X_, &Y_);
    //cout<<"acc  ="<<acc<<endl;
    CHECK_vector_equal_test_success<double>({0.6}, {acc}, __FUNCTION__);
}


void test_MNIST_read(){
    vector<double>labels; 
    read_Mnist_Labels("../../data/mnist/train-labels-idx1-ubyte", labels); 
    for (auto iter = labels.begin(); iter != labels.end(); iter++) 
    {
        if(iter > labels.begin() + 9) break;
        cout << *iter << " ";
    } 


    cout<<endl;
    vector<double> images;
    read_Mnist_Images("../../data/mnist/t10k-images-idx3-ubyte", images);
    for (int i = 0; i < images.size(); i++)
    {
        cout << images[i] << " ";  
        if(i > 9) break;
    }
    cout<<endl<<" success finished "<<__FUNCTION__<<endl;
}



void test_label_2_one_hot(){
    int data_num = 5,   class_num = 3;
    vector<double> label = {0,1,2,0,2};
    vector<double> label_one_hot(data_num * class_num);
    vector<double> expect_vec = {1,0,0, 0,1,0, 0,0,1, 1,0,0, 0,0,1};
    label_2_one_hot(&label.front(), &label_one_hot.front(), data_num, class_num);
    CHECK_vector_equal_test_success(expect_vec, label_one_hot, __FUNCTION__);
}



void test_compute_dropout_forward_and_backward(){
    MD_Vec<double> bs_in, bs_out, _;
    double dropout_ratio = 0.8;
    vector<double> keep_mask = {0,1,0,0,0,0};
    get_softmax_test_data(bs_in, bs_out, _);
    vector<double>expect_bs_in_data  = bs_in.data;
    vector<double>expect_bs_in_diff  = {0,  bs_out.diff[1] *5    ,0,0,0,0};
    vector<double>expect_bs_out_data = {0,  bs_in.data[1]  *5    ,0,0,0,0};
    vector<double>expect_bs_out_diff = bs_out.diff;


    compute_dropout<double>(&bs_in, &bs_out, dropout_ratio, &keep_mask.front(), true);
    compute_dropout_backward<double>(&bs_in, &bs_out, dropout_ratio, &keep_mask.front(), true);

    CHECK_vector_equal_test_success_by_diff<double>(expect_bs_in_data, bs_in.data,      any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success_by_diff<double>(expect_bs_in_diff, bs_in.diff,      any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success_by_diff<double>(expect_bs_out_data, bs_out.data,    any2str(__FILE__) +  " :line: " + any2str(__LINE__));
    CHECK_vector_equal_test_success_by_diff<double>(expect_bs_out_diff, bs_out.diff,    __FUNCTION__);
}




int main(){
    test_CHECK_equal();
    test_get_type_str();
    test_compute_softmax_and_softmax_loss_forward_backward();
    test_compute_fc_by_mat_mul_forward_and_backward();
    test_math_RELU_forward_and_backward();
    test_compute_conv2d_by_mat_mul_backward();
    test_img2col_by_kernelmat();
    test_col2img_by_kernelmat();
    test_compute_conv2d_by_7_for();
    test_compute_conv2d_by_mat_mul();
    test_transpose_matrix();
    test_split_string();
    test_any2str();
    test_load_txt_to_vector_string();
    test_mat_mul();
    test_vec_address();
    test_init_by_MDVec();
    test_get_random_Labels();
    test_copy_vector_from_to();
    test_get_rand_vec();
    test_mat_mul_run_time();
    test_get_random_data_and_label();
    test_compute_accuracy();
    test_MNIST_read();
    test_label_2_one_hot();
    test_compute_dropout_forward_and_backward();
}




