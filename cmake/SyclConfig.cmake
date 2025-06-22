message("   ---- SYCL config section ----")
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)


message(STATUS "Configure SYCL backend")
message(STATUS "Chosen SYCL implementation : ACPP")
set(SYCL_FLAGS "")

include(AcppConfig)
message( " ---- SYCL backend config ---- ")
message( "  SYCL_COMPILER : ${SYCL_COMPILER}")

message( " -------------------------------------- ")
message(STATUS "Configure SYCL backend - done")