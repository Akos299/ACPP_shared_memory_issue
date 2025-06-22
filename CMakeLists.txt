cmake_minimum_required(VERSION 3.12)


project(acpp-shared-memory-issue)

# configure SYCL 
set(CMAKE_MODULES_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
find_package(AdaptiveCpp CONFIG REQUIRED)

# Set build type to Release if unset
if(CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE was not set, setting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Set global flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build flags
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG -fno-omit-frame-pointer" CACHE STRING "Flags used by the C++ compiler during debug builds." FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math" CACHE STRING "Flags used by the C++ compiler during release builds." FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -DNDEBUG -march=native -ffast-math -g -fno-omit-frame-pointer" CACHE STRING "Flags used by the C++ compiler during release builds with debug info." FORCE)



#Set generator 
if(CMAKE_GENERATOR STREQUAL "Ninja" AND
 ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9) OR 
    (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.5)))

    # Force colored warning in Ninja's output, if the compiler has -fdiagnostics-color support.
    # Rationale in htpps://github.com/ninja-build/ninja/issues/814
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostic-color=always")
    set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -fdiagnostics-color=always")
endif()

set(supported_implementations
	AdaptiveCpp
)

list(FIND supported_implementations ${SYCL_IMPL} impl_idx)

if(NOT SYCL_IMPL OR impl_idx EQUAL -1)
	message(FATAL_ERROR "Please specify SYCL_IMPL (one of: ${supported_implementations})")
endif()

if(SYCL_IMPL STREQUAL "AdaptiveCpp")
	find_package(AdaptiveCpp REQUIRED)
endif()

set(my_kernels src/saxpy/saxpy.cpp
)

foreach(kernel IN LISTS my_kernels)
	get_filename_component(target ${kernel} NAME_WE)

	add_executable(${target} ${kernel})

	if(SYCL_IMPL STREQUAL "AdaptiveCpp")
		add_sycl_to_target(TARGET ${target} SOURCES ${kernel})
	endif()

  install(TARGETS ${target} RUNTIME DESTINATION bin/my_kernels/)
  get_filename_component(dir ${kernel} DIRECTORY)
  set_property(TARGET ${target} PROPERTY FOLDER ${dir})
endforeach(kernel)


# Set debug level 
# if(NOT ACPP_DEBUG_LEVEL)
#     if(CMAKE_BUILD_TYPE MATCHES "Debug")
#         set(ACPP_DEBUG_LEVEL 3 CACHE STRING "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)" FORCE)
#     else()
#         set(ACPP_DEBUG_LEVEL 2 CACHE STRING "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)" FORCE)
#     endif()
# endif()




#OMP_ROOT
if(APPLE)
    if(NOT DEFINED OpenMP_ROOT AND NOT DEFINED OMP_ROOT)
        execute_process(COMMAND brew list libomp
        COMMAND grep libomp.a
        COMMAND sed -E "s/\\/lib\\/.*//"
        OUTPUT_VARIABLE DefaultOMP_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE)

        #
        set(OpenMP_ROOT ${DefaultOMP_ROOT})
        set(OMP_ROOT ${DefaultOMP_ROOR})
    endif()  
endif()



#Use add_definitions for now and for old cmake versions
#cmake_policy(SET CMP0005 NEW)
#add_definitions(-DHIPSYCL_DEBUG_LEVEL=${ACPP_DEBUG_LEVEL})

#include_directories(${ACPP_BINARY_ROOT} ${ACPP_SOURCE_ROOT})

#add_subdirectory(src/saxpy)



