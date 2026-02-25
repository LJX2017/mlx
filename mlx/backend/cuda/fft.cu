// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <cufft.h>
#include <nvtx3/nvtx3.hpp>

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mlx::core {

namespace cu {

template <typename T>
__global__ void scale_fft_output(T* data, int64_t size, float scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size) {
    data[index] = data[index] * scale;
  }
}

} // namespace cu

namespace {

class CufftPlan {
 public:
  CufftPlan() {
    CHECK_CUFFT_ERROR(cufftCreate(&handle_));
  }

  ~CufftPlan() {
    if (handle_ != 0) {
      cufftDestroy(handle_);
    }
  }

  CufftPlan(const CufftPlan&) = delete;
  CufftPlan& operator=(const CufftPlan&) = delete;

  operator cufftHandle() const {
    return handle_;
  }

 private:
  cufftHandle handle_{0};
};

void execute_cufft_last_axis(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    bool inverse) {
  auto in_length = static_cast<long long>(in.shape(-1));
  auto out_length = static_cast<long long>(out.shape(-1));
  auto batch = static_cast<long long>(in.size() / in_length);
  auto out_batch = static_cast<long long>(out.size() / out_length);
  if (batch != out_batch) {
    throw std::runtime_error(
        "[FFT] Unexpected batch size mismatch between input and output.");
  }

  cufftType transform_type;
  long long n_fft = 0;
  if (in.dtype() == complex64 && out.dtype() == complex64) {
    transform_type = CUFFT_C2C;
    n_fft = in_length;
    if (in_length != out_length) {
      throw std::runtime_error("[FFT] Unexpected C2C input and output lengths.");
    }
  } else if (in.dtype() == float32 && out.dtype() == complex64) {
    transform_type = CUFFT_R2C;
    n_fft = in_length;
    if (out_length != (in_length / 2 + 1)) {
      throw std::runtime_error("[FFT] Unexpected R2C input and output lengths.");
    }
  } else if (in.dtype() == complex64 && out.dtype() == float32) {
    transform_type = CUFFT_C2R;
    n_fft = out_length;
    if (in_length != (out_length / 2 + 1)) {
      throw std::runtime_error("[FFT] Unexpected C2R input and output lengths.");
    }
  } else {
    throw std::runtime_error(
        "[FFT] Received unexpected input and output type combination.");
  }

  CufftPlan plan;
  CHECK_CUFFT_ERROR(cufftSetAutoAllocation(plan, 0));

  long long n[] = {n_fft};
  long long inembed[] = {in_length};
  long long outembed[] = {out_length};
  size_t workspace_size = 0;
  CHECK_CUFFT_ERROR(cufftMakePlanMany64(
      plan,
      /* rank= */ 1,
      n,
      inembed,
      /* istride= */ 1,
      /* idist= */ in_length,
      outembed,
      /* ostride= */ 1,
      /* odist= */ out_length,
      transform_type,
      batch,
      &workspace_size));

  auto* workspace = allocate_workspace(encoder, workspace_size);
  CHECK_CUFFT_ERROR(cufftSetWorkArea(plan, workspace));
  CHECK_CUFFT_ERROR(cufftSetStream(plan, encoder.stream()));

  encoder.set_input_array(in);
  encoder.set_output_array(out);

  auto capture = encoder.capture_context();
  if (transform_type == CUFFT_C2C) {
    auto* in_ptr = reinterpret_cast<cufftComplex*>(
        const_cast<complex64_t*>(gpu_ptr<complex64_t>(in)));
    auto* out_ptr = reinterpret_cast<cufftComplex*>(gpu_ptr<complex64_t>(out));
    CHECK_CUFFT_ERROR(cufftExecC2C(
        plan, in_ptr, out_ptr, inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
  } else if (transform_type == CUFFT_R2C) {
    auto* in_ptr = const_cast<float*>(gpu_ptr<float>(in));
    auto* out_ptr = reinterpret_cast<cufftComplex*>(gpu_ptr<complex64_t>(out));
    CHECK_CUFFT_ERROR(cufftExecR2C(plan, in_ptr, out_ptr));
  } else {
    auto* in_ptr = reinterpret_cast<cufftComplex*>(
        const_cast<complex64_t*>(gpu_ptr<complex64_t>(in)));
    auto* out_ptr = gpu_ptr<float>(out);
    CHECK_CUFFT_ERROR(cufftExecC2R(plan, in_ptr, out_ptr));
  }
}

float inverse_scale(
    const array& in,
    const array& out,
    const std::vector<size_t>& axes) {
  if (axes.empty()) {
    return 1.0f;
  }
  const auto& shape = out.dtype() == float32 ? out.shape() : in.shape();
  double n = 1.0;
  for (auto axis : axes) {
    n *= shape[axis];
  }
  return 1.0f / static_cast<float>(n);
}

void scale_inverse_fft(cu::CommandEncoder& encoder, array& out, float scale) {
  if (out.size() == 0 || scale == 1.0f) {
    return;
  }

  auto n = static_cast<int64_t>(out.size());
  constexpr uint32_t block_size = 256;
  auto grid_size = static_cast<uint32_t>((n + block_size - 1) / block_size);

  encoder.set_input_array(out);
  encoder.set_output_array(out);

  if (out.dtype() == float32) {
    encoder.add_kernel_node(
        cu::scale_fft_output<float>,
        dim3(grid_size),
        dim3(block_size),
        0,
        gpu_ptr<float>(out),
        n,
        scale);
  } else if (out.dtype() == complex64) {
    encoder.add_kernel_node(
        cu::scale_fft_output<cu::complex64_t>,
        dim3(grid_size),
        dim3(block_size),
        0,
        gpu_ptr<cu::complex64_t>(out),
        n,
        scale);
  } else {
    throw std::runtime_error("[FFT] Unsupported dtype for inverse scaling.");
  }
}

array fft_op_axis(
    const array& in,
    size_t axis,
    bool inverse,
    bool real,
    size_t inverse_real_output_length,
    cu::CommandEncoder& encoder,
    const Stream& s,
    std::vector<array>& temporaries) {
  auto fft_input = in;
  int axis_int = static_cast<int>(axis);
  int last_axis = in.ndim() - 1;
  bool transpose = axis_int != last_axis;

  if (transpose) {
    fft_input = swapaxes_in_eval(fft_input, axis_int, last_axis);
  }
  if (!fft_input.flags().row_contiguous) {
    auto contiguous = contiguous_copy_gpu(fft_input, s);
    temporaries.push_back(contiguous);
    fft_input = contiguous;
  }

  auto out_shape = fft_input.shape();
  if (real) {
    out_shape.back() =
        inverse ? inverse_real_output_length : out_shape.back() / 2 + 1;
  }

  auto out_dtype = (real && inverse) ? float32 : complex64;
  array fft_output(out_shape, out_dtype, nullptr, {});
  fft_output.set_data(cu::malloc_async(fft_output.nbytes(), encoder));
  temporaries.push_back(fft_output);
  execute_cufft_last_axis(encoder, fft_input, fft_output, inverse);

  if (!transpose) {
    return fft_output;
  }

  auto swapped = swapaxes_in_eval(fft_output, axis_int, fft_output.ndim() - 1);
  array restored(swapped.shape(), swapped.dtype(), nullptr, {});
  copy_gpu(swapped, restored, CopyType::General, s);
  temporaries.push_back(restored);
  return restored;
}

} // namespace

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("FFT::eval_gpu");
  if (out.size() == 0) {
    return;
  }

  assert(inputs.size() == 1);
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  if (axes_.empty()) {
    out.copy_shared_buffer(inputs[0]);
    return;
  }

  auto current = inputs[0];
  std::vector<array> temporaries;
  int n_axes = static_cast<int>(axes_.size());

  for (int i = n_axes - 1; i >= 0; --i) {
    int reverse_index = n_axes - i - 1;
    int index = inverse_ ? reverse_index : i;
    size_t axis = axes_[index];
    bool step_real = real_ && index == (n_axes - 1);

    size_t inverse_real_output_length = 0;
    if (step_real && inverse_) {
      inverse_real_output_length = out.shape(static_cast<int>(axis));
    }

    current = fft_op_axis(
        current,
        axis,
        inverse_,
        step_real,
        inverse_real_output_length,
        encoder,
        s,
        temporaries);
  }

  out.copy_shared_buffer(current);
  if (inverse_) {
    scale_inverse_fft(encoder, out, inverse_scale(inputs[0], out, axes_));
  }

  for (auto& arr : temporaries) {
    encoder.add_temporary(arr);
  }
}

} // namespace mlx::core
