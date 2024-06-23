#!/bin/bash

# 保存当前工作目录
ORIGINAL_DIR=$(pwd)

# 错误处理函数
error_exit() {
    echo "Error: $1"
    exit 1
}

gcc --version || error_exit "Failed to check gcc version"
g++ --version || error_exit "Failed to check g++ version"
gfortran --version || error_exit "Failed to check gfortran version"
cmake --version || error_exit "Failed to check cmake version"

# OpenBLAS 安装 —— LAPACK 和 Armadillo 库的前置
wget https://mirrors.aliyun.com/macports/distfiles/OpenBLAS/OpenBLAS-0.3.25.tar.gz || error_exit "Failed to download OpenBLAS"
tar zxvf OpenBLAS-0.3.25.tar.gz || error_exit "Failed to extract OpenBLAS"
cd OpenBLAS-0.3.25/
mkdir build && cd build

cmake .. || error_exit "Failed to configure OpenBLAS"
cmake --build . --parallel $(nproc) || error_exit "Failed to build OpenBLAS"
cmake --build . --target install || error_exit "Failed to install OpenBLAS"

cd $ORIGINAL_DIR

# LAPACK 安装 —— 作为 Armadillo 库的前置
wget https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz || error_exit "Failed to download LAPACK"
tar zxvf v3.12.0.tar.gz || error_exit "Failed to extract LAPACK"
cd lapack-3.12.0
mkdir build
cd build

cmake .. || error_exit "Failed to configure LAPACK"
cmake --build . --parallel $(nproc) || error_exit "Failed to build LAPACK"
cmake --build . --target install || error_exit "Failed to install LAPACK"

cd $ORIGINAL_DIR

# Armadillo 库安装
wget https://sourceforge.net/projects/arma/files/armadillo-12.8.4.tar.xz || error_exit "Failed to download Armadillo"
xz -d armadillo-12.8.4.tar.xz || error_exit "Failed to decompress Armadillo"
tar -xf armadillo-12.8.4.tar || error_exit "Failed to extract Armadillo"
cd armadillo-12.8.4
mkdir build
cd build

cmake -D CMAKE_EXE_LINKER_FLAGS="-lpthread -lgfortran -pthread" .. || error_exit "Failed to configure Armadillo"
cmake --build . --parallel $(nproc) || error_exit "Failed to build Armadillo"
cmake --build . --target install || error_exit "Failed to install Armadillo"

cd $ORIGINAL_DIR

# boost库安装
wget https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz || error_exit "Failed to download Boost"
tar zxvf boost_1_85_0.tar.gz || error_exit "Failed to extract Boost"
cd boost_1_85_0
./bootstrap.sh || error_exit "Failed to bootstrap Boost"
sudo ./b2 install || error_exit "Failed to install Boost"
cd $ORIGINAL_DIR

# oneTBB库安装
wget https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2020.3.tar.gz || error_exit "Failed to download oneTBB"
tar zxvf v2020.3.tar.gz || error_exit "Failed to extract oneTBB"
cd oneTBB-2020.3
make -j8 || error_exit "Failed to build oneTBB"
BUILD_DIR=$(find build -type d -name "linux*release")
echo "Generated build directory: $BUILD_DIR"
mkdir lib
cp $BUILD_DIR/* lib/ || error_exit "Failed to copy oneTBB build files"
cd $ORIGINAL_DIR

# tiff库安装 —— 这个是 proj库的前置
wget https://download.osgeo.org/libtiff/tiff-4.6.0.tar.gz || error_exit "Failed to download libtiff"
tar -zxvf tiff-4.6.0.tar.gz || error_exit "Failed to extract libtiff"
cd tiff-4.6.0
mkdir -p build
cd build

cmake -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libtiff-4.6.0 \
      -DCMAKE_INSTALL_PREFIX=/usr -G Ninja .. || error_exit "Failed to configure libtiff"
ninja || error_exit "Failed to build libtiff"
ninja test || error_exit "Failed to test libtiff"
ninja install || error_exit "Failed to install libtiff"
cd $ORIGINAL_DIR

# 安装 sqlite3 —— 这个是proj库的前置
wget https://www.sqlite.org/2024/sqlite-src-3460000.zip || error_exit "Failed to download SQLite"
unzip sqlite-src-3460000.zip || error_exit "Failed to extract SQLite"
cd sqlite-src-3460000/
mkdir build
cd build

../configure --prefix=/usr/local/sqlite3460000 --enable-retree=yes || error_exit "Failed to configure SQLite"
make || error_exit "Failed to build SQLite"
make install || error_exit "Failed to install SQLite"
mv /usr/bin/sqlite3 /usr/bin/sqlite3_old || error_exit "Failed to move old SQLite binary"
ln -s /usr/local/sqlite3460000/bin/sqlite3 /usr/bin/sqlite3 || error_exit "Failed to link new SQLite binary"

cd $ORIGINAL_DIR

# Proj 库安装 —— 这个是 GDAL库的前置
wget https://download.osgeo.org/proj/proj-9.4.0.tar.gz || error_exit "Failed to download Proj"
tar zxvf proj-9.4.0.tar.gz || error_exit "Failed to extract Proj"
cd proj-9.4.0
mkdir build
cd build
cmake -D CMAKE_EXE_LINKER_FLAGS="-lpthread -pthread" .. || error_exit "Failed to configure Proj"
cmake --build . --parallel $(nproc) || error_exit "Failed to build Proj"
cmake --build . --target install || error_exit "Failed to install Proj"
cd $ORIGINAL_DIR

# GDAL 库安装 —— 雷达图像处理库
wget https://github.com/OSGeo/gdal/releases/download/v3.9.0/gdal-3.9.0.tar.gz || error_exit "Failed to download GDAL"
tar zxvf gdal-3.9.0.tar.gz || error_exit "Failed to extract GDAL"
cd gdal-3.9.0
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release || error_exit "Failed to configure GDAL"
cmake --build . --parallel $(nproc) || error_exit "Failed to build GDAL"
cmake --build . --target install || error_exit "Failed to install GDAL"

cd $ORIGINAL_DIR

# 提示编译完成
echo "所有项目编译完成！"