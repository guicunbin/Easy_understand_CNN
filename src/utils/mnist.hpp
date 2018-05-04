#ifndef _MNIST_HPP_
#define _MNIST_HPP_

#include "../headers/headers.hpp"
#include "./base_utils.hpp"
using namespace std;  
  

int ReverseInt(int i)  
{
    unsigned char ch1, ch2, ch3, ch4;  
    ch1 = i & 255;  
    ch2 = (i >> 8) & 255;  
    ch3 = (i >> 16) & 255;  
    ch4 = (i >> 24) & 255;  
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;  
}  
  


void read_Mnist_Labels(string filename, vector<double>&labels)
{
    ifstream file(filename, ios::binary);  
    if (file.is_open())  
    {  
        int magic_number = 0;  
        int number_of_images = 0;  
        file.read((char*)&magic_number, sizeof(magic_number));  
        file.read((char*)&number_of_images, sizeof(number_of_images));  
        magic_number = ReverseInt(magic_number);  
        number_of_images = ReverseInt(number_of_images);  
        cout << "magic number = " << magic_number << endl;  
        cout << "number of images = " << number_of_images << endl;  
          
        labels.resize(number_of_images * 10, 0);
        vector<double> old_labels(number_of_images, 0);
        for (int i = 0; i < number_of_images; i++)  
        {  
            unsigned char label = 0;  
            file.read((char*)&label, sizeof(label));  
            old_labels[i] = (double)label;
        }
        label_2_one_hot(&old_labels.front(),    &labels.front(),    number_of_images,  10);
    }
}
  



void read_Mnist_Images(string filename, vector<double>&images)
{  
    ifstream file(filename, ios::binary);  
    if (file.is_open())  
    {  
        int magic_number = 0;  
        int number_of_images = 0;  
        int n_rows = 0;  
        int n_cols = 0;  
        unsigned char label;  
        file.read((char*)&magic_number, sizeof(magic_number));  
        file.read((char*)&number_of_images, sizeof(number_of_images));  
        file.read((char*)&n_rows, sizeof(n_rows));  
        file.read((char*)&n_cols, sizeof(n_cols));  
        magic_number = ReverseInt(magic_number);  
        number_of_images = ReverseInt(number_of_images);  
        n_rows = ReverseInt(n_rows);  
        n_cols = ReverseInt(n_cols);  
  
        cout << "magic number = " << magic_number << endl;  
        cout << "number of images = " << number_of_images << endl;  
        cout << "rows = " << n_rows << endl;  
        cout << "cols = " << n_cols << endl;  
 

        images.resize(number_of_images * n_cols * n_rows, 0);
        double *p = &images.front();
        for (int i = 0; i < number_of_images * n_cols * n_rows; i++)
        {
            unsigned char image = 0;
            file.read((char*)&image, sizeof(image));
            *(p++) = double(image) * 0.00390625;
        }
    }
}
  




#endif
