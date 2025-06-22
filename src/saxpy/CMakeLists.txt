add_executable(saxpy saxpy.cpp)
target_compile_options(saxpy PUBLIC ${SYCL_FLAGS})
target_link_options(saxpy PUBLIC ${SYCL_FLAGS})