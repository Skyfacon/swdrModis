# CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目信息
project(MyProject VERSION 1.0)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 包含头文件目录
include_directories(include)

# 查找源文件
file(GLOB SOURCES "src/*.cpp")

# 添加可执行文件
add_executable(MyProject ${SOURCES})

# 查找并链接 Boost 库
find_package(Boost REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(MyProject ${Boost_LIBRARIES})
endif()

# 查找并链接 TBB 库
find_package(TBB REQUIRED)
if(TBB_FOUND)
    include_directories(${TBB_INCLUDE_DIRS})
    target_link_libraries(MyProject ${TBB_LIBRARIES})
endif()

# 查找并链接 Armadillo 库
find_package(Armadillo REQUIRED)
if(Armadillo_FOUND)
    include_directories(${Armadillo_INCLUDE_DIRS})
    target_link_libraries(MyProject ${Armadillo_LIBRARIES})
endif()

# 查找并链接 GDAL 库
find_package(GDAL REQUIRED)
if(GDAL_FOUND)
    include_directories(${GDAL_INCLUDE_DIRS})
    target_link_libraries(MyProject ${GDAL_LIBRARIES})
endif()

# 打印库路径以进行调试
message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
message(STATUS "TBB include dirs: ${TBB_INCLUDE_DIRS}")
message(STATUS "Armadillo include dirs: ${Armadillo_INCLUDE_DIRS}")
message(STATUS "GDAL include dirs: ${GDAL_INCLUDE_DIRS}")
