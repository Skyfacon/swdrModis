# CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目信息
project(MyProject VERSION 1.0)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
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


# TBB 库的链接还是有问题，可以参照下 TBB库 cmake下的 README
# list(APPEND CMAKE_MODULE_PATH "/root/swdrmodis/oneTBB-2020.3/cmake/")
#set(CMAKE_PREFIX_PATH "/root/oneTBB-2020.3")
# list(APPEND CMAKE_MODULE_PATH "/root/oneTBB-2020.3/cmake/")
# set(TBB_DIR "/root/oneTBB-2020.3/cmake")
set(TBB_DIR "/root/swdrModis-main/oneTBB-2020.3/cmake")
find_package(TBB REQUIRED)
if(TBB_FOUND)
    include_directories(${TBB_INCLUDE_DIRS})
    target_link_libraries(MyProject TBB::tbb)
else()
    message(WARNING "TBB not found")
endif()

# 查找并链接 Armadillo 库
find_package(Armadillo REQUIRED)
if(Armadillo_FOUND)
    include_directories(${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(MyProject ${ARMADILLO_LIBRARIES})
    target_link_libraries(MyProject ${ARMADILLO_LIBRARIES} gfortran)
else()
    message(WARNING "Armadillo not found")
endif()

# 查找并链接 GDAL 库
find_package(GDAL REQUIRED)
if(GDAL_FOUND)
    include_directories(${GDAL_INCLUDE_DIRS})
    target_link_libraries(MyProject ${GDAL_LIBRARIES})
endif()


# 手动链接pthread库
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(MyProject Threads::Threads)



# # 打印库路径以进行调试
# message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
# message(STATUS "Armadillo include dirs: ${ARMADILLO_INCLUDE_DIRS}")
# message(STATUS "GDAL include dirs: ${GDAL_INCLUDE_DIRS}")
# message(STATUS "OpenBLAS include dirs: ${OpenBLAS_INCLUDE_DIRS}")
# message(STATUS "TBB include dirs: ${TBB_INCLUDE_DIRS}")
# message(STATUS "LAPACK include dirs: ${LAPACK_INCLUDE_DIRS}")


# 打印 Armadillo 库信息
message(STATUS "Armadillo version: ${ARMADILLO_VERSION}")
message(STATUS "Armadillo libraries: ${ARMADILLO_LIBRARIES}")
message(STATUS "Armadillo include dirs: ${ARMADILLO_INCLUDE_DIRS}")

# 打印 TBB 库信息
message(STATUS "TBB version: ${TBB_VERSION}")
message(STATUS "TBB include dirs: ${TBB_INCLUDE_DIRS}")
message(STATUS "TBB libraries: ${TBB_LIBRARIES}")



# 打印 GDAL库信息
message(STATUS "GDAL version: ${GDAL_VERSION}")
message(STATUS "GDAL include dirs: ${GDAL_INCLUDE_DIRS}")
message(STATUS "GDAL libraries: ${GDAL_LIBRARIES}")