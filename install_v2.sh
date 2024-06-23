#!/bin/bash

# 更新系统
sudo yum update -y

# 安装必要的软件包
sudo yum -y groupinstall 'Development Tools'
sudo yum -y install wget gmp-devel mpfr-devel libmpc-devel

# 安装 gcc-c++ 10.2.0
wget https://ftp.gnu.org/gnu/gcc/gcc-10.2.0/gcc-10.2.0.tar.gz
tar -xzf gcc-10.2.0.tar.gz
cd gcc-10.2.0

./contrib/download_prerequisites

mkdir build
cd build
../configure --enable-languages=c,c++ --disable-multilib
make -j$(nproc)
sudo make install


# 这块有问题，还不如编译安装 cmake 和 g++，不要用这个 devtoolset-10-toolchain
# sudo yum install -y centos-release-scl
# sudo yum install -y devtoolset-10-toolchain
sudo yum install -y wget ninja-build.x86_64 xz.x86_64 libcurl-devel.x86_64 tcl.x86_64 libsqlite3x-devel.x86_64 cmake3.x86_64 libgfortran5.x86_64


ln -s /usr/bin/cmake3 /usr/bin/cmake

# 提示用户执行下一个脚本
echo "Setup complete. running build.sh..."

# 启用 devtoolset-10 并执行 build.sh
./build.sh