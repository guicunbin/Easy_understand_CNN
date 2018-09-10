#ifndef _BASE_UTILS_HPP_
#define _BASE_UTILS_HPP_

#include"../headers/headers.hpp"


template <typename Dtype>
string get_type_str(Dtype obj){
    string s = typeid(obj).name();
    return s;
}






void throw_error(string commment){
    cout<<commment<<endl;
    throw commment;
}





template <typename Dtype>
void copy_value_from_to(Dtype *A, Dtype *B, int N){
    Dtype * A_end = A + N;
    while(A < A_end){
        //cout<<*A<<","<<*B<<endl;
        //cout<<A<<","<<B<<endl;
        *B = *A;
        B++;    A++;
    }
}


template <typename Dtype>
string any2str(Dtype num){
    // string  to  str also   supported
    std::ostringstream ss;
	ss << num;
	std::string s(ss.str());
	return s;
    //  return std::to_string(num);
}


template <typename Dtype>
void CHECK_equal(Dtype a,  Dtype b, string commment = ""){
    // support:  char string float double .......... float* double*  int* ........
    if(a == b) return;
    throw_error("CHECK_equal error !!!  " + commment + ":  "+any2str<Dtype>(a)+" != "+any2str<Dtype>(b));
}


template <typename Dtype>
void CHECK_less(Dtype a,  Dtype b, string commment = ""){
   if(a >= b){
    throw_error("CHECK_less error !!!  " + commment +": "+ any2str<Dtype>(a)+" >= "+any2str<Dtype>(b));
  }
}



template <typename Dtype>
void label_2_one_hot(Dtype *old, Dtype* one_hot, int data_num, int class_num){
    Dtype *p_old = old;
    Dtype *p_one = one_hot;
    while(p_old < old + data_num){
        for(int i=0;    i<class_num;   i++,  p_one ++){
            if(*p_old == i)     *p_one = 1;
            else                *p_one = 0;
        }
        p_old ++;
    }
}





























template <typename Dtype>
void copy_vector_from_to(vector<Dtype>  &A_vec,     vector<Dtype> &B_vec, int start_index_A, int end_index_A){
    //cout<<&A_vec.back()<<","<<&B_vec.back()<<endl;
    CHECK_less(start_index_A, end_index_A,      "start less than end");
    CHECK_less(start_index_A, int(A_vec.size()),"start less than size_A");
    int NA = int(A_vec.size());
    if(end_index_A > NA ){
        copy_value_from_to<Dtype>(&A_vec[0] + start_index_A,  &B_vec[0],  NA - start_index_A);
        
        copy_value_from_to<Dtype>(&A_vec[0],  &B_vec[0] + NA - start_index_A,   end_index_A - NA);
    }
    else{
        copy_value_from_to<Dtype>(&A_vec[0] + start_index_A,  &B_vec[0],  end_index_A - start_index_A);
    }
}




template <typename T> 
void release_vector(vector<T>& nums){
	nums.clear();
	nums.shrink_to_fit();
}



void a_mul_X_plus_Y(const int N, const float alpha, const float* X,  float* Y){ 
    cblas_saxpy(N, alpha, X, 1, Y, 1); 
}

void a_mul_X_plus_Y(const int N, const double alpha, const double* X,  double* Y){ 
    cblas_daxpy(N, alpha, X, 1, Y, 1); 
}







template <typename Dtype>
void CHECK_equal_by_diff(Dtype a,  Dtype b, string commment = "", double diff = 1e-5){
    // not support Pi, Pf (int*,  float* .........).................
    if(get_type_str(a)[0]=='P') throw_error(" not support this Dtype : " + get_type_str(a));
    if(a == b) return;
    if(double(a) - diff < double(b) && double(b) < double(a) + diff) return;
    throw_error("CHECK_equal error !!!  " + commment + ":  "+any2str<Dtype>(a)+" != "+any2str<Dtype>(b));
}






template <typename Dtype>
void CHECK_vector_equal(vector<Dtype> vec1,  vector<Dtype> vec2, string commment=""){
    CHECK_equal(int(vec1.size()), int(vec2.size()), " check_vector_size ");
    int n = vec1.size();
    Dtype* p1 = &vec1[0], *p2 = &vec2[0];
    for(int i=0; i<n; i++){
        CHECK_equal(*(p1++), *(p2++), commment);
    }
}




template <typename Dtype>
void CHECK_vector_equal_by_diff(vector<Dtype> vec1,  vector<Dtype> vec2){
    CHECK_equal(int(vec1.size()), int(vec2.size()), " check_vector_size ");
    int n = vec1.size();
    Dtype* p1 = &vec1[0], *p2 = &vec2[0];
    for(int i=0; i<n; i++){
        CHECK_equal_by_diff(*(p1++), *(p2++));
    }
}




template <typename Dtype>
void print_vec(vector<Dtype> nums, string commment = "", int max_num = INT_MAX){
    if(commment.size() != 0){
        cout<<endl<<commment;
    }
    cout<<endl<<" [ ";
    for(int i=0; i<nums.size() && i<max_num; i++){
        cout<<nums[i]<<",";
    }
    cout<<" ] "<<endl;
}



template <typename Dtype>
void CHECK_vector_equal_test_success(vector<Dtype> expect_vec, vector<Dtype> output, string function_str){
    cout<<" start CHECK "<<function_str<<" .... "<<endl;

    CHECK_vector_equal<Dtype>(expect_vec, output);

    cout<<endl<<" ===> function: "<<function_str<<" success finished !!"<<endl<<endl;
}



template <typename Dtype>
void CHECK_vector_equal_test_success_by_diff(vector<Dtype> expect_vec, vector<Dtype> output, string function_str){
    cout<<" start CHECK "<<function_str<<" .... "<<endl;

    CHECK_vector_equal_by_diff<Dtype>(expect_vec, output);

    cout<<endl<<" ===> function: "<<function_str<<" success finished !!"<<endl<<endl;
}







template <typename Dtype>
class MD_Vec {
	public:
        // use one Dim  vector to save muti dim vector;   can ensure the address continue
		typedef Dtype value_type;
		vector<Dtype> data;
        vector<Dtype> diff;
		vector<int>  shape;
        Dtype learn_rate = 0;


		MD_Vec(vector<int> sha = {1,1,1}, Dtype val=0){
			shape = sha;
			int len = 1;
			for(int i=0; i<shape.size(); i++){ len *= shape[i]; }
			data.resize(len,    Dtype(val));
            diff.resize(len,    Dtype(val));
		}



        void set_diff_to_zero(){
            fill_n(&diff[0],    this->count(),  Dtype(0));
        }


        void Update(){
            CHECK_less(Dtype(0), learn_rate, " learn_rate must be larger than 0");
            a_mul_X_plus_Y(this->count(), - learn_rate, &diff[0],  &data[0]);
            //for(int i=0; i<this->count();   i++){
            //    data[i] -= learn_rate * diff[i];
            //}
            set_diff_to_zero();
        }



        void release(){
            release_vector<Dtype>(data);
            release_vector<Dtype>(diff); 
            release_vector<int>(shape); 
        }


        int count(int start=0, int end = INT_MAX){
            CHECK_less(-1, start);
            end = min(int(shape.size()),  end);
            int cnt = 1;
            for(int i=start;  i<end;    i++){
                cnt  *=  shape[i];
            }
            return cnt;
        }




        void print(string commment = ""){
            cout<<" -----------------------------------------------"<<endl;
            cout<<" "<<commment<<endl;
            print_vec<int>(shape,   " shape = ");
            print_vec<Dtype>(data,    " data  = ");
            print_vec<Dtype>(diff,    " diff  = ");
            cout<<" -----------------------------------------------"<<endl;
        }


        void print_shape(){
            cout<<" -----------------------------------------------"<<endl;
            print_vec<int>(shape,   " shape = ");
            cout<<" -----------------------------------------------"<<endl;
        }

        void init_by_shape_and_constant(vector<int> shap, Dtype val =0){
            shape = shap;
            int len = 1;
            for(int i=0;    i<shape.size();     len *= shape[i], i++);
            data.resize(len, val);
            diff.resize(len, val);
        }


        void copy_to(MD_Vec<Dtype> &output){
            CHECK_vector_equal(output.shape,  shape);
            CHECK_equal(int(output.data.size()), int(data.size()));
            //copy data and diff
            Dtype * pdata = &data[0], * pdata_end = &data.back() + 1,  *pdata_out = &output.data[0];
            Dtype * pdiff = &diff[0], * pdiff_end = &diff.back() + 1,  *pdiff_out = &output.diff[0];
            for(; pdata<pdata_end;  pdata ++, pdiff ++, pdata_out ++,   pdiff_out ++){
                *pdata_out = *pdata;
                *pdiff_out = *pdiff;
            }
        }


        void init_by_MDVec(MD_Vec<Dtype> & input){
            this->init_by_shape_and_constant(input.shape, 0);        
            input.copy_to(*this);
        }


        void CHECK_EQ(MD_Vec<Dtype> &top){
            CHECK_vector_equal(shape,  top.shape,   "CHECK_EQ shape : ");
            CHECK_vector_equal(data,   top.data,    "CHECK_EQ data  : ");
            CHECK_vector_equal(diff,   top.diff,    "CHECK_EQ diff  : ");
        }
};








template <typename Dtype>
void transpose_matrix(vector<Dtype>&nums,  vector<int> shape){
    if(shape.size()!=2) throw_error(" must be 2dim vector");
    if(shape[0] == shape[1]){
        for(int i=0; i<shape[0]; i++){
            for(int j=0; j<i; j++){
                //swap(nums[i][j], nums[j][i]);
                swap(nums[i * shape[1] + j], nums[j*shape[1] + i]);
            }
        }
    }
    else{
        vector<Dtype> new_nums(shape[0]*shape[1]);
        for(int i=0; i<shape[1]; i++){
            for(int j=0; j<shape[0]; j++){
                //swap(nums[i][j], nums[j][i]);
                new_nums[i * shape[0] + j] = nums[j * shape[1] + i];
            }
        }
        nums = new_nums;
    }
}




vector<int> stl_sort(vector<int> &nums){
    sort(nums.begin(), nums.end());
    return nums;
}


struct ListNode{
    int val;
    ListNode * next;
    ListNode(int i=0){
        val = i; 
        //next = nullptr;
        next = NULL;
    }
};


template <typename T> 
void delete_pointer(T*& a){  
    delete a;  
    a = NULL;  
}




void release_listnode_pointers(ListNode* head){
	if(!head) return;
	ListNode* tmp = head;
	while(head->next){
		tmp = head;
		head = head->next;
		free(tmp);
	};
	free(head);
}


template <typename T> 
bool is_be_sorted(vector<T> nums){
    for(int i=1; i<nums.size(); i++){
        if(nums[i-1] > nums[i]) return false;
    }
    return true;
}




template <typename Dtype>
void print_c_array(Dtype nums[], int n){
    cout<<endl<<" { ";
    for(int i=0; i<n; i++){
        cout<<nums[i]<<",";
    }
    cout<<" } "<<endl;
}




void print_listnodes(ListNode* head){
    ListNode *p = head;
    cout<<endl<<" { ";
    while(p){
        cout<<p->val<<",";
        p = p->next;
    }
    cout<<" } "<<endl;
}






template <typename Dtype>
vector<Dtype> get_rand_vec(int test_size, Dtype mean = 0, Dtype std = 0.01){
    std::default_random_engine generator;
    std::normal_distribution<Dtype> distribution(mean, std);
    vector<Dtype> test_vec(test_size, 0);
    //  for(int i=0; i<test_size; test_vec[i] = Dtype(i),i++);
    for(int i=0; i<test_size; test_vec[i] = distribution(generator), i++);
    //  for(int i=0; i<test_size; test_vec[i] = rand(),i++);
    //  random_shuffle(test_vec.begin(), test_vec.end());
    return test_vec;
}


template <typename Dtype>
void get_bernoulli_vec(int test_size, Dtype p, Dtype *data){
    std::default_random_engine generator;
    std::bernoulli_distribution distribution(p);
    for (int i=0; i<test_size; data[i] = Dtype(distribution(generator)), i++);
}





template <typename Dtype>
void get_rand_c_array(Dtype* test_vec, int test_size){
    //for(int i=0; i<test_size; test_vec[i]=rand() % 10,i++);
    for(int i=0; i<test_size; test_vec[i]=(rand() % 100) / 10.0,i++);
    if(test_size < 50){
        print_c_array(test_vec, test_size);
    }
}



void print_time_using(clock_t start, string commment){
    cout<<commment<<(double(clock() - start)*1000) / CLOCKS_PER_SEC <<" ms"<<endl; 
}




void print_sort_time(vector<int> sort_func(vector<int>& ), vector<int> test_vec, bool is_print_vec){
    if (is_print_vec){
        cout<<"original = "<<endl; 
        print_vec(test_vec);
    }

    clock_t start = clock();
    //here the assignment is very Time-consuming; 1.5 % time  for bucket_sort
    //test_vec = sort_func(test_vec);
    sort_func(test_vec);
    print_time_using(start, "----------------------------------------------------------------- run_time = ");

    if (is_print_vec){
        cout<<"finally = "<<endl; 
        print_vec(test_vec);
        //test_vec = {0,2,1};
        if(is_be_sorted(test_vec)){
            cout<<"--- YES --- "<<endl;
        }
        else{
            throw std::invalid_argument("-- not sorted ---");
        }
    }
}



void load_txt_to_vector_string(const char *txt_path, vector<string>& vec_str){
    fstream fst;
    //fst.open(txt_path, ios::in | ios::out);
    fst.open(txt_path, ios::in);
    string line_str;
    while(std::getline(fst, line_str)){
        vec_str.push_back(line_str);
    }
    fst.close();
}









template <typename Dtype>
void mat_mul_use_cblas(Dtype *A, Dtype *B, Dtype *C, int row_op_A, int col_op_B, int col_op_A, bool transpose_A=false, bool transpose_B=false, Dtype alpha = Dtype(0))
{
// --------------------------------------------------------------------
// http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html#gaeda3cbd99c8fb834a60a6412878226e1
    CBLAS_TRANSPOSE TA = transpose_A ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE TB = transpose_B ? CblasTrans : CblasNoTrans;
	int lda = (TA == CblasNoTrans) ? col_op_A : row_op_A;  // first dim of the A;   first dim mean the col_op_A
  	int ldb = (TB == CblasNoTrans) ? col_op_B : col_op_A;  // first dim of the B  
    //int ldc = (TB == CblasNoTrans) ? col_op_B : row_op_A;
    int ldc = col_op_B;
    cblas_dgemm(CblasRowMajor, TA, TB, row_op_A, col_op_B, col_op_A,  1 , A, lda, B, ldb, alpha, C, ldc);

}




template <typename Dtype>
void mat_mul_self_implement_1(Dtype* A, Dtype* B, Dtype* C, int row_A, int col_B, int col_A, bool row_major = true){
    // just static Dtype res[100]; can return res; http://blog.51cto.com/forlinux/1530404
    Dtype tmp;
    for(int i=0; i<row_A; i++){
        for(int j=0; j<col_B; j++){
            tmp = 0;
            for(int k1=0; k1<col_A; k1++){
                tmp += (*(A + i*col_A + k1)) * (*(B + k1*col_B + j));
            }
            *(C + i*col_B + j) = tmp;
        }
    }
}




template <typename Dtype>
void mat_mul_self_implement_2(Dtype* A, Dtype* B, Dtype* C,  int row_A, int col_B, int col_A, bool row_major = true){
    Dtype tmp;
    Dtype *pa = A, *pb = B, *pc = C;
    for(int i=0; i<row_A; i++){
        for(int j=0; j<col_B; j++){
            tmp = 0; // use tmp 可以提高缓存命中率,更快
            //C[i * col_B + j] = 0;
            for(int k1=0; k1<col_A; k1++){
                //tmp += A[i*col_A + k1] * B[k1 * col_B + j];
                tmp += *(pa + k1) * B[k1 * col_B + j];
                //C[i * col_B + j] += A[i*col_A + k1]*B[k1 * col_B + j];
            }
            //C[i * col_B + j] = tmp;
            *(pc + j) = tmp;
            //pc[j] = tmp;
        }
        pa += col_A;
        pc += col_B;
    }
}







vector<string> split_string(string st,  string sep){
    //seq empty
    if(sep.empty()){
        vector<string> res(st.size());
        string tmp = "";    int i=0;
        for(string::iterator it = st.begin(); it != st.end(); it++){
            res[i++] = tmp + *it;
        }
        return res;
    }
    //seq.size() >= 1;
    int start=0;
    string tmp = "";
    int n1 = st.size(),  n2 = sep.size();
    vector<string> res;
    for(int i=0; i<n1; i++){
        if(sep[0]==st[i]){
            //start 
            bool is_sep = true;
            int j=0;
            for(; j<n2; j++, i++){
                if(sep[j] != st[i])
                    is_sep = false;
            }
            // finish 
            if(is_sep){
                res.push_back(st.substr(start, i-n2-start));
                start = i;
            }
            // need backtrack
            else{
                i = i-j;
            }
        }
    }
    if(start < n1){
        res.push_back(st.substr(start, n1-start));
    }
    return res;
}





template <typename Dtype>
void img2col_by_kernelmat(Dtype* p_vec3d, Dtype* p_vec2d, vector<int> vec3d_shape,  vector<int> vec2d_shape, int kernel_size, int stride, int pad = 0){
    int C_in = vec3d_shape[0], H_in = vec3d_shape[1], W_in = vec3d_shape[2];
    int H_out = (H_in - kernel_size + 2*pad) / stride + 1; 
    int W_out = (W_in - kernel_size + 2*pad) / stride + 1; 
    int H_res = C_in * kernel_size * kernel_size;
    int W_res = H_out * W_out;
    CHECK_equal(H_res, vec2d_shape[0], any2str(__LINE__));
    CHECK_equal(W_res, vec2d_shape[1], any2str(__LINE__));

    Dtype *p_2d = p_vec2d;
    Dtype *p_3d  = p_vec3d;

    for(int c_in=0; c_in<C_in; c_in++){
        for(int k_h=0; k_h<kernel_size; k_h++){
            for(int k_w=0; k_w<kernel_size; k_w++){
            // start one row of the return vec2d;
                for(int h_out=0, h_in = k_h - pad; h_out<H_out; h_out++, h_in += stride){
                    for(int w_out=0, w_in = k_w - pad; w_out<W_out; w_out++, w_in += stride){
                        if(h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in){
                            *p_2d = 0;
                        }
                        *p_2d = p_3d[h_in* W_in + w_in];
                        p_2d ++;
                    }
                }
            //end one row of the return vec2d;
            }
        }
        // travel finished one channel
        p_3d += H_in * W_in;
    }
}





template <typename Dtype>
void col2img_by_kernelmat(Dtype* p_vec3d, Dtype* p_vec2d, vector<int> vec3d_shape,  vector<int> vec2d_shape, int kernel_size, int stride, int pad = 0){
    int C_in = vec3d_shape[0], H_in = vec3d_shape[1], W_in = vec3d_shape[2];
    int H_out = (H_in - kernel_size + 2*pad) / stride + 1; 
    int W_out = (W_in - kernel_size + 2*pad) / stride + 1; 
    int H_res = C_in * kernel_size * kernel_size;
    int W_res = H_out * W_out;
    CHECK_equal(H_res, vec2d_shape[0], any2str(__LINE__));
    CHECK_equal(W_res, vec2d_shape[1], any2str(__LINE__));

    Dtype *p_2d = p_vec2d;
    Dtype *p_3d  = p_vec3d;

    for(int c_in=0; c_in<C_in; c_in++){
        for(int k_h=0; k_h<kernel_size; k_h++){
            for(int k_w=0; k_w<kernel_size; k_w++){
            // start one row of the return vec2d;
                for(int h_out=0, h_in = k_h - pad; h_out<H_out; h_out++, h_in += stride){
                    for(int w_out=0, w_in = k_w - pad; w_out<W_out; w_out++, w_in += stride){
                        if(h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in){
                            *p_2d = 0;
                        }
                        //  *p_2d = p_3d[h_in* W_in + w_in];
                        //  p_2d ++;
                        //  notice: this is add to p_3d; because the diff need add 
                        p_3d[h_in * W_in + w_in] += *p_2d;
                        p_2d ++;
                    }
                }
            //end one row of the return vec2d;
            }
        }
        // travel finished one channel
        p_3d += H_in * W_in;
    }
}




template <typename Dtype>
vector<Dtype>  get_random_Labels(vector<int> shape){
    CHECK_equal(int(shape.size()),  2,  any2str(__LINE__));
    vector<Dtype> Labels(shape[0]*shape[1], 0);


    Dtype *p_Labels = &Labels[0], * p_Labels_end = &Labels[shape[0]*shape[1] - 1] + 1;
    while(p_Labels < p_Labels_end){
        //cout<<p_Labels<<","<<p_Labels_end<<endl;
        int groud_truth = rand() % shape[1];
        for(int j=0; j<shape[1]; j++){
            //cout<<"j = "<<j<<endl;
            if(j == groud_truth){
                *(p_Labels +j) = 1; break;
            }
        }
        p_Labels += shape[1];
    }
    cout<<"end this function :"<<__FUNCTION__<<endl;
    return Labels;
}





template <typename Dtype>
void get_random_data_and_label(vector<Dtype> &X, vector<Dtype> &Y,  int data_num,
        int feature_num,    int class_num){
    std::default_random_engine generator;
    CHECK_equal(int(X.size()),  data_num * feature_num);
    CHECK_equal(int(Y.size()),  data_num * class_num);
    vector<Dtype> mean_vec = get_rand_vec<Dtype>(class_num, 100, 50);
    Dtype std = 0.1;
    CHECK_equal(data_num % class_num, 0);
    int data_num_each_class = data_num / class_num;
    vector<pair<vector<Dtype>, Dtype>>  XYS(data_num);;
    for(int y=0; y<class_num; y++){
        for(int i=0;    i<data_num_each_class;  i++){
            XYS[y * data_num_each_class + i] = make_pair(get_rand_vec<Dtype>(feature_num, mean_vec[y], std),  Dtype(y));
        }
    }
    random_shuffle(XYS.begin(), XYS.end());
    fill_n(Y.begin(),   data_num * class_num,   Dtype(0));
    for(int i=0;    i<data_num;     i++){
        for(int j=0;    j<feature_num;  j++){
            X[i*feature_num + j] = XYS[i].first[j];
        }
        Y[i*class_num + int(XYS[i].second)] = Dtype(1);
    }
    release_vector<pair<vector<Dtype>, Dtype>>(XYS);
}






#endif
