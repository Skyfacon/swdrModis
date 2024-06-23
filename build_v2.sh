# 保存当前工作目录
ORIGINAL_DIR=$(pwd)

source /opt/rh/devtoolset-10/enable

gcc --version
g++ --version
gfortran --version
cmake --version

# OpenBLAS 安装 —— LAPACK 和 Armadillo 库的前置
wget https://mirrors.aliyun.com/macports/distfiles/OpenBLAS/OpenBLAS-0.3.25.tar.gz
tar zxvf OpenBLAS-0.3.25.tar.gz
cd OpenBLAS-0.3.25/
mkdir build && cd build

cmake ..
cmake --build . --parallel $(nproc)
cmake --build . --target install 

cd $ORIGINAL_DIR

# LAPACK 安装 —— 作为 Armadillo 库的前置
wget https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz

tar zxvf v3.12.0.tar.gz

cd lapack-3.12.0


mkdir build

cd build

cmake ..

cmake --build . --parallel $(nproc)

cmake --build . --target install

cd $ORIGINAL_DIR

# Armadillo 库安装
wget https://sourceforge.net/projects/arma/files/armadillo-12.8.4.tar.xz

xz -d armadillo-12.8.4.tar.xz

tar -xf armadillo-12.8.4.tar

cd armadillo-12.8.4

mkdir build

cd build


cmake -D CMAKE_EXE_LINKER_FLAGS="-lpthread -lgfortran -pthread" ..

cmake --build . --parallel $(nproc)

cmake --build . --target install

cd $ORIGINAL_DIR


# boost库安装
wget https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz # 还是下的太慢了
tar zxvf boost_1_85_0.tar.gz
cd boost_1_85_0
./bootstrap.sh
sudo ./b2 install
cd $ORIGINAL_DIR

# oneTBB库安装
wget https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2020.3.tar.gz
tar zxvf v2020.3.tar.gz
cd oneTBB-2020.3
make -j8
BUILD_DIR=$(find build -type d -name "linux*release")
echo "Generated build directory: $BUILD_DIR"
mkdir lib
cp $BUILD_DIR/* lib/
cd $ORIGINAL_DIR

# tiff库安装 —— 这个是 proj库的前置
wget https://download.osgeo.org/libtiff/tiff-4.6.0.tar.gz

tar -zxvf tiff-4.6.0.tar.gz 

cd tiff-4.6.0

mkdir -p build 
cd  build 

cmake -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libtiff-4.6.0 \
      -DCMAKE_INSTALL_PREFIX=/usr -G Ninja .. && ninja

ninja test
ninja install 
cd $ORIGINAL_DIR

# 安装 sqlite3 —— 这个是proj库的前置
wget https://www.sqlite.org/2024/sqlite-src-3460000.zip

unzip sqlite-src-3460000.zip

cd sqlite-src-3460000/

mkdir build

cd build

../configure --prefix=/usr/local/sqlite3460000 --enable-retree=yes

make && make install

mv /usr/bin/sqlite3 /usr/bin/sqlite3_old

ln -s /usr/local/sqlite3460000/bin/sqlite3 /usr/bin/sqlite3

cd $ORIGINAL_DIR


# Proj 库安装 —— 这个是 GDAL库的前置
wget https://download.osgeo.org/proj/proj-9.4.0.tar.gz
tar zxvf proj-9.4.0.tar.gz 
cd proj-9.4.0
mkdir build
cd build
cmake ..
cmake --build . --parallel $(nproc)
cmake --build . --target install
cd $ORIGINAL_DIR

# GDAL 库安装 —— 雷达图像处理库
wget https://github.com/OSGeo/gdal/releases/download/v3.9.0/gdal-3.9.0.tar.gz

tar zxvf gdal-3.9.0.tar.gz

cd gdal-3.9.0

mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)
cmake --build . --target install

cd $ORIGINAL_DIR


# 提示编译完成
echo "所有项目编译完成！"