#!/bin/bash

# 更新系统
sudo yum update -y

# 安装必要的软件包
sudo yum -y groupinstall 'Development Tools'

sudo yum install -y epel-release
sudo yum install -y centos-release-scl
sudo yum install -y devtoolset-10-toolchain
sudo yum install -y wget ninja-build.x86_64 xz.x86_64 libcurl-devel.x86_64 tcl.x86_64 unzip.x86_64 python27-python-devel.x86_64 cmake3.x86_64  # libsqlite3x-devel.x86_64


ln -s /usr/bin/cmake3 /usr/bin/cmake

# 提示用户执行下一个脚本
echo "Setup complete. Enabling devtoolset-10 and running build.sh..."

# 启用 devtoolset-10 并执行 build.sh
source /opt/rh/devtoolset-10/enable
./build.sh