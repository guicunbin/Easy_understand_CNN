## if you don't have blas , you need add blas_path to your bashrc


cd ./dependency/OpenBLAS_build/
blas_path=$(pwd)


echo -e "\n\n" >> ~/.bashrc
echo "export PATH="\$PATH:${blas_path}/bin/"" >> ~/.bashrc
echo "export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:${blas_path}/lib/"" >> ~/.bashrc
echo "export CPLUS_INCLUDE_PATH="\$CPLUS_INCLUDE_PATH:${blas_path}/include/"" >> ~/.bashrc
echo "export C_INCLUDE_PATH="\$C_INCLUDE_PATH:${blas_path}/include/"" >> ~/.bashrc
echo "export LIBRARY_PATH="\$LIBRARY_PATH:${blas_path}/lib/"" >> ~/.bashrc
