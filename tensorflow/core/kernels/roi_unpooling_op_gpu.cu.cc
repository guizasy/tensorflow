#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>

#include "tensorflow/core/kernels/roi_unpooling_op_gpu.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace {

template <typename T>
__global__ void ROIUnpoolForward(
    const int nthreads, const T* bottom_data,
    const int num_rois, const float spatial_scale,
    const int height, const int width, const int channels, 
    const int pooled_height, const int pooled_width, T* top_data,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, h, w, c) coords in bottom data
    int n = index;
    int c = n % channels;
    n /= channels;
    int w = n % width;
    n /= width;
    int h = n % height;
    n /= height;

    T value = 0;
    // Accumulate value over all ROIs that has this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const T* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_w = static_cast<int>(round(offset_bottom_rois[1] * spatial_scale));
      int roi_start_h = static_cast<int>(round(offset_bottom_rois[2] * spatial_scale));
      int roi_end_w = static_cast<int>(round(offset_bottom_rois[3] * spatial_scale));
      int roi_end_h = static_cast<int>(round(offset_bottom_rois[4] * spatial_scale));

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int offset = roi_n * pooled_height * pooled_width * channels;
      const T* offset_bottom_data = bottom_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      T bin_size_h = static_cast<T>(roi_height)
                         / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width)
                         / static_cast<T>(pooled_width);

      // find the pool regions that covers the current location
      int phstart = static_cast<int>(floor(static_cast<T>(h - roi_start_h) / bin_size_h));
      int phend = static_cast<int>(ceil(static_cast<T>(h - roi_start_h + 1) / bin_size_h));
      int pwstart = static_cast<int>(floor(static_cast<T>(w - roi_start_w) / bin_size_w));
      int pwend = static_cast<int>(ceil(static_cast<T>(w - roi_start_w + 1) / bin_size_w));

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), height);
          hend = min(max(hend + roi_start_h, 0), height);
          wstart = min(max(wstart + roi_start_w, 0), width);
          wend = min(max(wend + roi_start_w, 0), width);

          const T this_size = static_cast<T>((hend - hstart) * (wend - wstart));
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          if (is_empty) {
            continue;
          }

          value += offset_bottom_data[(ph * pooled_width + pw) * channels + c] / this_size;
        }
      }
    }

    top_data[index] = value;
  }
}

template <typename T>
__global__ void ROIUnpoolBackward(
    const int nthreads, const T* top_diff,
    const float spatial_scale, const int height, const int width, 
    const int channels, const int pooled_height, const int pooled_width,
    const T* bottom_rois, T* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, ph, pw, c) is an element in the pooled output
    int n = index;
    int c = n % channels;
    n /= channels;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;

    const T* bottom_rois_this = bottom_rois + n * 5;
    int roi_batch_ind = static_cast<int>(bottom_rois_this[0]);
    int roi_start_w = static_cast<int>(round(bottom_rois_this[1] * spatial_scale));
    int roi_start_h = static_cast<int>(round(bottom_rois_this[2] * spatial_scale));
    int roi_end_w = static_cast<int>(round(bottom_rois_this[3] * spatial_scale));
    int roi_end_h = static_cast<int>(round(bottom_rois_this[4] * spatial_scale));

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));
    // Add roi offsets and clip to input boundaries

    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);

    const T this_size = static_cast<T>((hend - hstart) * (wend - wstart));
    bool is_empty = (hend <= hstart) || (wend <= wstart);
    if (is_empty) {
      return;
    }
    const T* top_diff_this = top_diff + roi_batch_ind * channels * height * width;

    // average pooling the region
    T value = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int top_diff_index = (h * width + w) * channels + c;
        value += top_diff_this[top_diff_index] / this_size;
      }
    }

    bottom_diff[index] = value;
  }
}

} // namespace

template <typename T>
bool RoiUnpoolForwardGPU(
    const T* bottom_data, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const T* bottom_rois,
    T* top_data, const Eigen::GpuDevice& d) {

  const int output_size = batch_size * height * width * channels;
  CudaLaunchConfig config = GetCudaLaunchConfig(output_size, d);

  ROIUnpoolForward<T>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        output_size, bottom_data,
        num_rois, spatial_scale,
        height, width, channels, 
        pooled_height, pooled_width, top_data,
        bottom_rois);

  return d.ok();
}

#define DECLARE_GPU_SPEC(T)                                                                     \
  template bool RoiUnpoolForwardGPU(const T* bottom_data, const float spatial_scale,            \
                                    const int batch_size, const int num_rois,                   \
                                    const int height, const int width, const int channels,      \
                                    const int pooled_height, const int pooled_width,            \
                                    const T* bottom_rois, T* top_data,                          \
                                    const Eigen::GpuDevice& d);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC

template <typename T>
bool RoiUnpoolBackwardGPU(
    const T* top_diff, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const T* bottom_rois,
    T* bottom_diff, const Eigen::GpuDevice& d) {

  const int output_size = num_rois * pooled_height * pooled_width * channels;
  CudaLaunchConfig config = GetCudaLaunchConfig(output_size, d);

  ROIUnpoolBackward<T>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        output_size, top_diff,
        spatial_scale, height, width, 
        channels, pooled_height, pooled_width,
        bottom_rois, bottom_diff);

  return d.ok();
}

#define DECLARE_GPU_SPEC(T)                                                                       \
  template bool RoiUnpoolBackwardGPU(const T* top_diff, const float spatial_scale,                \
                                     const int num_rois, const int height,                        \
                                     const int width, const int channels,                         \
                                     const int pooled_height, const int pooled_width,             \
                                     const T* bottom_rois, T* bottom_diff,                        \
                                     const Eigen::GpuDevice& d);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
