// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/dtype_utils.h"

#include <fmt/format.h>
#include <cuda/cmath>
#include <vector>

namespace mlx::core {

void check_cublas_error(const char* name, cublasStatus_t err) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    // TODO: Use cublasGetStatusString when it is widely available.
    throw std::runtime_error(
        fmt::format("{} failed with code: {}.", name, static_cast<int>(err)));
  }
}

void check_cufft_error(const char* name, cufftResult err) {
  if (err != CUFFT_SUCCESS) {
    auto err_str = "Unknown error";
    switch (err) {
      case CUFFT_INVALID_PLAN:
        err_str = "CUFFT_INVALID_PLAN";
        break;
      case CUFFT_ALLOC_FAILED:
        err_str = "CUFFT_ALLOC_FAILED";
        break;
      case CUFFT_INVALID_TYPE:
        err_str = "CUFFT_INVALID_TYPE";
        break;
      case CUFFT_INVALID_VALUE:
        err_str = "CUFFT_INVALID_VALUE";
        break;
      case CUFFT_INTERNAL_ERROR:
        err_str = "CUFFT_INTERNAL_ERROR";
        break;
      case CUFFT_EXEC_FAILED:
        err_str = "CUFFT_EXEC_FAILED";
        break;
      case CUFFT_SETUP_FAILED:
        err_str = "CUFFT_SETUP_FAILED";
        break;
      case CUFFT_INVALID_SIZE:
        err_str = "CUFFT_INVALID_SIZE";
        break;
      case CUFFT_UNALIGNED_DATA:
        err_str = "CUFFT_UNALIGNED_DATA";
        break;
#ifdef CUFFT_INCOMPLETE_PARAMETER_LIST
      case CUFFT_INCOMPLETE_PARAMETER_LIST:
        err_str = "CUFFT_INCOMPLETE_PARAMETER_LIST";
        break;
#endif
#ifdef CUFFT_INVALID_DEVICE
      case CUFFT_INVALID_DEVICE:
        err_str = "CUFFT_INVALID_DEVICE";
        break;
#endif
#ifdef CUFFT_PARSE_ERROR
      case CUFFT_PARSE_ERROR:
        err_str = "CUFFT_PARSE_ERROR";
        break;
#endif
#ifdef CUFFT_NO_WORKSPACE
      case CUFFT_NO_WORKSPACE:
        err_str = "CUFFT_NO_WORKSPACE";
        break;
#endif
#ifdef CUFFT_NOT_IMPLEMENTED
      case CUFFT_NOT_IMPLEMENTED:
        err_str = "CUFFT_NOT_IMPLEMENTED";
        break;
#endif
#ifdef CUFFT_NOT_SUPPORTED
      case CUFFT_NOT_SUPPORTED:
        err_str = "CUFFT_NOT_SUPPORTED";
        break;
#endif
      default:
        break;
    }
    throw std::runtime_error(fmt::format("{} failed: {}.", name, err_str));
  }
}

void check_cuda_error(const char* name, cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error(
        fmt::format("{} failed: {}", name, cudaGetErrorString(err)));
  }
}

void check_cuda_error(const char* name, CUresult err) {
  if (err != CUDA_SUCCESS) {
    const char* err_str = "Unknown error";
    cuGetErrorString(err, &err_str);
    throw std::runtime_error(fmt::format("{} failed: {}", name, err_str));
  }
}

void check_cudnn_error(const char* name, cudnnStatus_t err) {
  if (err != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(
        fmt::format("{} failed: {}.", name, cudnnGetErrorString(err)));
  }
}

const char* dtype_to_cuda_type(const Dtype& dtype) {
  switch (dtype) {
    case bool_:
      return "bool";
    case int8:
      return "int8_t";
    case int16:
      return "int16_t";
    case int32:
      return "int32_t";
    case int64:
      return "int64_t";
    case uint8:
      return "uint8_t";
    case uint16:
      return "uint16_t";
    case uint32:
      return "uint32_t";
    case uint64:
      return "uint64_t";
    case float16:
      return "__half";
    case bfloat16:
      return "__nv_bfloat16";
    case float32:
      return "float";
    case float64:
      return "double";
    case complex64:
      return "mlx::core::cu::complex64_t";
    default:
      return "unknown";
  }
}

CudaGraph::CudaGraph(cu::Device& device) {
  device.make_current();
  CHECK_CUDA_ERROR(cudaGraphCreate(&handle_, 0));
}

void CudaGraph::end_capture(cudaStream_t stream) {
  CHECK_CUDA_ERROR(cudaStreamEndCapture(stream, &handle_));
}

void CudaGraphExec::instantiate(cudaGraph_t graph) {
  assert(handle_ == nullptr);
  CHECK_CUDA_ERROR(cudaGraphInstantiate(&handle_, graph, nullptr, nullptr, 0));
}

CudaStream::CudaStream(cu::Device& device) {
  device.make_current();
  CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&handle_, cudaStreamNonBlocking));
}

void* allocate_workspace(cu::CommandEncoder& encoder, size_t workspace_size) {
  if (workspace_size == 0) {
    return nullptr;
  }

  // Workspace allocation should not be captured.
#ifndef NDEBUG
  cudaStreamCaptureStatus status;
  CHECK_CUDA_ERROR(cudaStreamIsCapturing(encoder.stream(), &status));
  assert(status == cudaStreamCaptureStatusNone);
#endif

  // Ensure workspace is 256-byte aligned.
  int nbytes = cuda::ceil_div(workspace_size, 256) * 256;
  array workspace(cu::malloc_async(nbytes, encoder), {nbytes}, int8);
  encoder.add_temporary(workspace);
  return gpu_ptr<void>(workspace);
}

} // namespace mlx::core
