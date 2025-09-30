cmake_minimum_required(VERSION 3.10)
project(TASK_03 VERSION 0.1.0 LANGUAGES CXX)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 查找 Ceres Solver 包
find_package(Ceres REQUIRED)

# 包含 OpenCV 和 Ceres 的头文件目录
include_directories(${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

# 指定源文件目录
set(SOURCE_DIR "./src")

# 收集源文件目录中的所有源文件
aux_source_directory(${SOURCE_DIR} SRC)

# 添加可执行文件，包含收集到的源文件
add_executable(TASK_03 ${SRC})

# 链接 OpenCV 和 Ceres Solver 库到你的可执行文件
target_link_libraries(TASK_03 ${OpenCV_LIBS} ${CERES_LIBRARIES})