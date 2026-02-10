cmake_minimum_required(VERSION 3.18)
project(pioneerml_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Reduce Torch config warnings.
set(USE_KINETO OFF CACHE BOOL "" FORCE)
set(TORCH_USE_KINETO OFF CACHE BOOL "" FORCE)
set(TORCH_CUDA_ARCH_LIST "12.0" CACHE STRING "" FORCE)

if(DEFINED ENV{CUDA_HOME})
  set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}" CACHE PATH "" FORCE)
  set(nvtx3_DIR "$ENV{CUDA_HOME}/lib64/cmake/nvtx3" CACHE PATH "" FORCE)
endif()
if(DEFINED ENV{CONDA_PREFIX})
  set(CMAKE_BUILD_RPATH "$ENV{CONDA_PREFIX}/lib" CACHE STRING "" FORCE)
  set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib" CACHE STRING "" FORCE)
endif()

# LibTorch
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(CURL REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# nlohmann/json
find_package(nlohmann_json REQUIRED)

# Pull in dataloaders (submodule) unless an external build is provided.
option(PIONEERML_INFERENCE_USE_EXTERNAL_DATALOADERS "Link against an existing dataloaders build" OFF)
set(PML_DATALOADERS_DIR "${CMAKE_CURRENT_LIST_DIR}/../pioneerml_dataloaders")
if(PIONEERML_INFERENCE_USE_EXTERNAL_DATALOADERS)
  if(NOT PIONEERML_DATALOADERS_LIB)
    message(FATAL_ERROR "PIONEERML_DATALOADERS_LIB must be set when using external dataloaders.")
  endif()
  add_library(pioneerml_dataloaders SHARED IMPORTED)
  set_target_properties(pioneerml_dataloaders PROPERTIES IMPORTED_LOCATION "${PIONEERML_DATALOADERS_LIB}")
  target_include_directories(pioneerml_dataloaders INTERFACE "${PML_DATALOADERS_DIR}/include")
else()
  if(EXISTS "${PML_DATALOADERS_DIR}/CMakeLists.txt")
    set(PIONEERML_DATALOADERS_BUILD_BINDINGS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${PML_DATALOADERS_DIR} ${CMAKE_BINARY_DIR}/pioneerml_dataloaders)
  else()
    message(FATAL_ERROR "pioneerml_dataloaders not found at ${PML_DATALOADERS_DIR}")
  endif()
endif()

add_executable(pioneerml_inference
  src/pioneerml_inference/main.cpp
  src/pioneerml_inference/runner/group_classifier_runner.cpp
  src/pioneerml_inference/runner/group_classifier_event_runner.cpp
  src/pioneerml_inference/runner/group_splitter_runner.cpp
  src/pioneerml_inference/runner/group_splitter_event_runner.cpp
  src/pioneerml_inference/runner/event_splitter_event_runner.cpp
  src/pioneerml_inference/runner/endpoint_regressor_runner.cpp
  src/pioneerml_inference/runner/endpoint_regressor_event_runner.cpp
)

target_include_directories(pioneerml_inference
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(pioneerml_inference
  PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cupti
    CURL::libcurl
    nlohmann_json::nlohmann_json
    pioneerml_dataloaders
)

# Torch uses -D_GLIBCXX_USE_CXX11_ABI flag in its config
set_property(TARGET pioneerml_inference PROPERTY CXX_STANDARD 17)
