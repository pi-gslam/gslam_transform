cmake_minimum_required(VERSION 2.8)
include(PICMake)

list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -DHAS_TOON")
pi_add_target(gslam_transform SHARED src REQUIRED GSLAM Ceres)
pi_report_target()
