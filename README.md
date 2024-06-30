# swdrModis 简介
雷达图像处理

# 安装编译方式
1. 下载本代码仓的zip压缩文件(或者直接git clone也可)
2. 在linux机器的/root路径下解压缩（如果是git clone的话，在root路径下直接git clone即可）
3. cd swdrModis-main
4. chmod +x install.sh
5. chmod +x build.sh
6. ./install.sh
7. 等待编译完成，即可在build文件夹找到可执行的二进制文件MyProject

# 运行方式
1. 需要在data/lut下上传lut文件（文件名目前写死在了代码中，后续应该以参数的形式传入）
2. 把要处理的raster图像上传到data/inputdata下
3. 直接运行 ./Myproject
4. 输出图像能在 /data/output文件夹下找到


# Attention
1. 相关配置文件在config文件夹下
2. 该程序内存占用极高，如果使用并行库Tbb的话，需要斟酌运行机器内存，单张10M图像，处理过程中内存占用最高达到约60G
3. 该编译和运行目前仅在CentOs7.9上测试成功过，如果是其他Linux系统，有些库的编译可能需要适配