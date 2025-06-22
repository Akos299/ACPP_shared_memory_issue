
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  message(FATAL_ERROR
    "Acpp cmake integration requires Clang compiler, but ${CMAKE_CXX_COMPILER_ID} is used "
    "please set clang as cxx compiler "
    "-DCMAKE_CXX_COMPILER=<clang_path>")
endif()


if(NOT (DEFINED ACPP_TARGETS))

    message(FATAL_ERROR "no target are set for AdaptiveCpp: "
    "known valid targets are :
    omp (OpenMP backend)
    generic (Single pass compiler)
    spirv (generate spirV)
    cuda:sm_xx  (CUDA backend)
        where xx can be
        50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 90a
    hip-gfxXXX (HIP / ROCM backend)
        where XXX can be
        701, 801, 802, 803, 900, 906, 908, 1010, 1011, 1012, 1030, 1031

    please set -DACPP_TARGETS=<target_list>
    depending on the compiler version
    exemple : -DACPP_TARGETS=omp;cuda:sm_52
    ")
endif()

find_package(AdaptiveCpp CONFIG)
if(NOT AdaptiveCpp_FOUND)
    message(FATAL_ERROR
    "acpp can not be found")
else()
    set(ACPP_CLANG "${CMAKE_CXX_COMPILER}$")
endif()

set(SYCL_FLAGS "${SYCL_FLAGS} - DSYCL_COMP_ACPP")
set(SYCL_COMPILER "ACPP_CMAKE")

message(" ---- Acpp compiler cmake config ---- ")
message("  ACPP_TARGETS : ${ACPP_TARGETS}")
message(" ------------------------------------ ")