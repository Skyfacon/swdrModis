#!/bin/bash

source /opt/rh/devtoolset-10/enable

# 保存当前工作目录
ORIGINAL_DIR=$(pwd)

# 错误处理函数
error_exit() {
    echo "Error: $1"
    exit 1
}

# 检查命令是否存在并打印版本的函数
check_command() {
    local cmd=$1
    command -v $cmd >/dev/null 2>&1 || error_exit "Failed to check $cmd version"
    local version=$($cmd --version 2>&1 | head -n 1)
    echo "Using $cmd: $version"
}


# # 设置GCC和G++路径
# export PATH=/usr/local/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH


check_command gcc
check_command g++
check_command gfortran
check_command cmake
check_command wget
check_command tar
check_command unzip


# 安装函数模板
install_package() {
    local package_name=$1
    local download_url=$2
    local tar_options=$3
    local config_commands=$4

    if [ ! -f "${package_name}_downloaded" ]; then
        wget $download_url || error_exit "Failed to download ${package_name}"
        touch "${package_name}_downloaded"
    fi

    if [ ! -f "${package_name}_extracted" ]; then
        tar $tar_options || unzip "${package_name}" || error_exit "Failed to extract ${package_name}"
        touch "${package_name}_extracted"
    fi

    if [ ! -f "${package_name}_built" ]; then
        cd ${package_name}
        mkdir -p build && cd build
        eval "$config_commands" || error_exit "Failed to configure/build/install ${package_name}"
        cd $ORIGINAL_DIR
        touch "${package_name}_built"
    fi
}

# 追加简化后的CMake代码到CMakeLists.txt
append_cmake_code() {
    cmake_file="../CMakeLists.txt"
    cmake_code="
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(TARGETS
    gdalinfo
    ogr2ogr
    gdal_footprint
    gdal_viewshed
    gdaltindex
    gdal_rasterize
    gdalmanage
    gdaltransform
    gdalenhance
    gdal_create
    gdalsrsinfo
    gdaladdo
    gdal_translate
    ogrlineref
    ogrinfo
    gdalwarp
    gdalbuildvrt
    gdal_contour
    gdallocationinfo
    gnmmanage
    ogrtindex
    gdalmdiminfo
    nearblack
    gdalmdimtranslate
    sozip
    gdal_grid
    gdaldem
    gnmanalyse
    test_ogrsf
)

foreach(target \${TARGETS})
    target_link_libraries(\${target} PRIVATE Threads::Threads)
endforeach()
"

    echo "$cmake_code" >> "$cmake_file"
    echo "CMake code appended to $cmake_file"
}


# OpenBLAS 安装
install_package "OpenBLAS-0.3.25" \
    "https://mirrors.aliyun.com/macports/distfiles/OpenBLAS/OpenBLAS-0.3.25.tar.gz" \
    "zxvf OpenBLAS-0.3.25.tar.gz" \
    "cmake .. && cmake --build . --parallel \$(nproc) && cmake --build . --target install"

# LAPACK 安装
install_package "lapack-3.12.0" \
    "https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz" \
    "zxvf v3.12.0.tar.gz" \
    "cmake .. && cmake --build . --parallel \$(nproc) && cmake --build . --target install"

# Armadillo 安装
install_package "armadillo-12.8.4" \
    "https://sourceforge.net/projects/arma/files/armadillo-12.8.4.tar.xz" \
    "Jxvf armadillo-12.8.4.tar.xz" \
    "cmake -DCMAKE_EXE_LINKER_FLAGS='-lpthread -lgfortran -pthread' .. && cmake --build . --parallel \$(nproc) && cmake --build . --target install"

# Boost 安装
install_package "boost-1.85.0" \
    "https://mirrors.aliyun.com/blfs/conglomeration/boost/boost-1.85.0-b2-nodocs.tar.xz" \
    "Jxvf boost-1.85.0-b2-nodocs.tar.xz" \
    "cd .. && ./bootstrap.sh && sudo ./b2 install --with-python include=/opt/rh/python27/root/usr/include/python2.7 library-path=/opt/rh/python27/root/usr/lib64"

# oneTBB 安装
install_package "oneTBB-2020.3" \
    "https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2020.3.tar.gz" \
    "zxvf v2020.3.tar.gz" \
    "cd .. && make -j8 && BUILD_DIR=\$(find build -type d -name 'linux*release') && mkdir -p lib/intel64/gcc4.8 && cp \$BUILD_DIR/* lib/intel64/gcc4.8/ && cmake -DTBB_ROOT=\$ORIGINAL_DIR/oneTBB-2020.3 -DTBB_OS=Linux -P cmake/tbb_config_generator.cmake"

# 
# cmake -DTBB_ROOT=${ORIGINAL_DIR}/oneTBB-2020.3 -DTBB_OS=Linux -P cmake/tbb_config_generator.cmake



# TIFF 安装
install_package "tiff-4.6.0" \
    "https://download.osgeo.org/libtiff/tiff-4.6.0.tar.gz" \
    "zxvf tiff-4.6.0.tar.gz" \
    "cmake -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libtiff-4.6.0 -DCMAKE_INSTALL_PREFIX=/usr -G Ninja .. && ninja && ninja test && ninja install"

# SQLite 安装 (再把头文件复制到 /usr/local/include, 把库文件复制到 /usr/local/lib64下即可)
install_package "sqlite-src-3460000" \
    "https://www.sqlite.org/2024/sqlite-src-3460000.zip" \
    "unzip sqlite-src-3460000.zip" \
    "../configure --prefix=/usr/local/sqlite3460000 --enable-rtree=yes && make && make install"

cp /usr/local/sqlite3460000/include/* /usr/local/include/
cp /usr/local/sqlite3460000/lib/* /usr/local/lib64/
rm -f /usr/bin/sqlite3
ln -fs /usr/local/sqlite3460000/bin/sqlite3 /usr/bin/sqlite3

echo "除了Proj 和 GDAL以外，所有项目编译完成！"

# Proj 安装
install_package "proj-9.4.0" \
    "https://download.osgeo.org/proj/proj-9.4.0.tar.gz" \
    "zxvf proj-9.4.0.tar.gz" \
    "cmake -DCMAKE_EXE_LINKER_FLAGS='-pthread' -DSQLITE3_DIR=/usr/local/sqlite3460000/ .. && cmake --build . --parallel \$(nproc) && cmake --build . --target install"

echo "Proj编译完成！"
# GDAL 安装
install_package "gdal-3.9.0" \
    "https://github.com/OSGeo/gdal/releases/download/v3.9.0/gdal-3.9.0.tar.gz" \
    "zxvf gdal-3.9.0.tar.gz" \
    "append_cmake_code && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --parallel \$(nproc) && cmake --build . --target install"

echo "GDAL编译完成！"


# 最后编译swdr项目
mkdir build && cd build
cmake ..
cmake --build .