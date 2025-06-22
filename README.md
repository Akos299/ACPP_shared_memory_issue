# ACPP_shared_memory_issue

## Usage

```
$ git clone https://github.com/Akos299/ACPP_shared_memory_issue.git

$ mkdir build && cd build
$ cmake -DSYCL_IMPL=[target SYCL implementation]  [other compiler arguments] ..
$ cmake --build .
$ make
```
Example compiling with CMake for AdaptiveCpp:
```
$ cmake -DSYCL_IMPL=AdaptiveCpp -DCMAKE_CXX_COMPILER=/path/to/llvm/build/bin/clang++ ..
```


