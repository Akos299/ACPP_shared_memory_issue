#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

template <typename T>
void usm_saxpy(sycl::queue &q, T a, T *x, T *y, size_t vec_size) {
          q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(vec_size), [=](sycl::id<1> id) {
              y[id] = a * x[id] + y[id];
            });
          });

  q.wait();
}


template<typename T>
void local_mem_saxpy(sycl::queue& q, T a , T*x, T*y, size_t vec_size, size_t group_size)
{
    const size_t group_count = (vec_size + group_size -1)/group_size;
    auto loc_mem_size = 2*group_size*sizeof(T);
    if(loc_mem_size > q.get_device().get_info<sycl::info::device::local_mem_size>()){
        std::cout << " not enough local memory. \n";
        std::cout << " loc_mem_size : "<< loc_mem_size <<" > max_loc_mem_size "<< q.get_device().get_info<sycl::info::device::local_mem_size>()<<"\n" ;
    }

    q.submit([&](sycl::handler&h){
        sycl::local_accessor<T> loc_acc_x(group_size, h);
        sycl::local_accessor<T> loc_acc_y(group_size, h);

        h.parallel_for(sycl::nd_range<1>(group_count*group_size, group_size), [=](sycl::nd_item<1> item)
        {
            size_t gid = item.get_global_id(0);
            size_t lid = item.get_global_id(1);
            
            if(gid < vec_size){
                loc_acc_x[lid] = x[gid];
                loc_acc_y[lid] = y[gid];
            }
        
            item.barrier(sycl::access::fence_space::local_space);

            if(gid < vec_size)
            {
                loc_acc_y[lid] = a * loc_acc_x[lid] + loc_acc_y[lid];
            }
            item.barrier(sycl::access::fence_space::local_space);

            if(gid < vec_size)
            {
                y[gid] = loc_acc_y[lid];
            }
            
        });
    
    });
    q.wait();

}

template <typename T>
void host_saxpy(T a, T*x, T* y, size_t vec_size)
{
    for(auto i = 0; i < vec_size; i++)
    {
        y[i] += a * x[i];
    }
}

void query_device_info(sycl::queue& q)

{
    auto name = q.get_device().get_info<sycl::info::device::name>();
    auto max_work_group_size = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    auto local_mem_type = q.get_device().get_info<sycl::info::device::local_mem_type>();
    std::cout << " name " << name <<  " max_work_group_size " << max_work_group_size << "\n";
    //  << " max_work_group_size " << max_work_group_size << " local_mem_type " << local_mem_type << "\n";
}

template <typename T>
void check_result(T* h_buf, T* d_buf, size_t vec_size)
{
    for(auto i = 0; i < vec_size; i++)
    {
        if(h_buf[i] != d_buf[i])
        {
            std::cout << " device saxpy is not correct ! \t at least the " << i <<"-th index values don't match : h_buf\
            ["<<i<<"] = " <<h_buf[i] << "and d_buf ["<<i <<"] = "<<d_buf[i]<< "\n" ;
            EXIT_FAILURE;
        }
        
    }
    std::cout << " host and device computation matches \n" ;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <vector_size> <group_size>\n";
        return 1;
    }

  size_t N = std::atoi(argv[1]);
  size_t group_size = std::atoi(argv[2]);
  float a = 0.5f;

  sycl::queue q{sycl::default_selector_v};
  query_device_info(q);

  float *x_d = sycl::malloc_device<float>(N, q);
  float *y_d = sycl::malloc_device<float>(N, q);
  float *x_h = new float[N];
  float *y_h = new float[N];
  float *z_h = new float[N];

  for (size_t i = 0; i < N; i++) {
    x_h[i] = 4.0;
    y_h[i] = 2.0;
  }

  q.memcpy(x_d, x_h, N * sizeof(float));
  q.memcpy(y_d, y_h, N * sizeof(float));
  q.wait();

//   std::cout << "------------ device[usm] saxpy -------------------" << "\n";
//   usm_saxpy<float>(q, a, x_d, y_d, N);
//   q.memcpy(z_h, y_d, N*sizeof(float)).wait();

  std::cout << "------------ host saxpy ---------------------" << "\n";
  host_saxpy<float>(a, x_h, y_h, N);
  
//   std::cout << "------------- check for [usm] -------------------------" << "\n";
//   check_result<float>(y_h, z_h, N);

  std::cout << "------------ device[local_mem] saxpy -------------------" << "\n";
  local_mem_saxpy<float>(q, a, x_d, y_d, N,group_size);
  q.memcpy(z_h, y_d, N*sizeof(float)).wait();
  std::cout << "------------- check for [local_mem] -------------------------" << "\n";
  check_result<float>(y_h, z_h, N);
 

  sycl::free(x_d, q);
  sycl::free(y_d, q);
  delete [] x_h;
  delete [] y_h;

  return 0;
}
