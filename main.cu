#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cub/device/device_batch_memcpy.cuh"


float get_max_bw(int dev = 0)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  return float(deviceProp.memoryBusWidth) * deviceProp.memoryClockRate * 2 / 8 / 1000 / 1000;
}


template <typename T>
thrust::host_vector<T> gen_uniform_buffer_sizes(std::size_t buffers,
                                                std::size_t buffer_size)
{
  thrust::host_vector<T> sizes(buffers, buffer_size);
  return sizes;
}

template <typename DataT, typename OffsetT>
class Input
{
  const std::size_t buffers {};
  std::size_t total_input_size {};

  thrust::device_vector<DataT> input;
  thrust::device_vector<DataT> output;

  thrust::device_vector<void*> in_pointers;
  thrust::device_vector<void*> out_pointers;

  thrust::device_vector<OffsetT> buffer_sizes;

public:
  Input(thrust::host_vector<OffsetT> h_buffer_sizes)
    : buffers(h_buffer_sizes.size())
    , in_pointers(buffers)
    , out_pointers(buffers)
  {
    total_input_size = thrust::reduce(h_buffer_sizes.begin(),
                                      h_buffer_sizes.end());

    input.resize(total_input_size);
    output.resize(total_input_size);

    thrust::host_vector<void*> h_in_pointers(buffers);
    thrust::host_vector<void*> h_out_pointers(buffers);

    DataT *in_ptr = thrust::raw_pointer_cast(input.data());
    DataT *out_ptr = thrust::raw_pointer_cast(output.data());

    for (std::size_t buffer = 0; buffer < buffers; buffer++)
    {
      h_in_pointers[buffer] = in_ptr;
      h_out_pointers[buffer] = out_ptr;

      in_ptr += h_buffer_sizes[buffer];
      out_ptr += h_buffer_sizes[buffer];

      h_buffer_sizes[buffer] *= sizeof(DataT);
    }

    in_pointers = h_in_pointers;
    out_pointers = h_out_pointers;
    buffer_sizes = h_buffer_sizes;
  }

  std::size_t get_bytes_read() const
  {
    return total_input_size * sizeof(DataT);
  }

  std::size_t get_bytes_written() const
  {
    return get_bytes_read();
  }

  void** get_input() const
  {
    return const_cast<void**>(thrust::raw_pointer_cast(in_pointers.data()));
  }

  void** get_output() const
  {
    return const_cast<void**>(thrust::raw_pointer_cast(out_pointers.data()));
  }

  const OffsetT* get_buffer_sizes() const
  {
    return thrust::raw_pointer_cast(buffer_sizes.data());
  }

  std::size_t get_num_buffers() const
  {
    return buffers;
  }

  float bytes_to_gb(std::size_t bytes) const
  {
    return static_cast<float>(bytes / 1024 / 1024 / 1024);
  }

  float get_bw(float ms) const
  {
    float seconds = ms / 1000.0f;
    return bytes_to_gb(get_bytes_read() + get_bytes_written()) / seconds;
  }
};


template <typename DataT,
          typename OffsetT>
void report_result(float ms, const Input<DataT, OffsetT> &input)
{
  const float achieved_bw = input.get_bw(ms);
  const float expected_bw = get_max_bw();

  std::cout << achieved_bw << " / " << expected_bw << " ("
            << (achieved_bw / expected_bw) * 100.0f << "%)" << std::endl;
}


template <typename DataT,
          typename OffsetT>
void measure_cub(const Input<DataT, OffsetT> &input)
{
  std::size_t temp_storage_bytes {};
  cub::DeviceBatchMemcpy(nullptr,
                         temp_storage_bytes,
                         input.get_input(),
                         input.get_output(),
                         input.get_buffer_sizes(),
                         input.get_num_buffers());

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  std::uint8_t *d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin);

  cub::DeviceBatchMemcpy(d_temp_storage,
                         temp_storage_bytes,
                         input.get_input(),
                         input.get_output(),
                         input.get_buffer_sizes(),
                         input.get_num_buffers());

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms {};
  cudaEventElapsedTime(&ms, begin, end);

  report_result(ms, input);

  cudaEventDestroy(end);
  cudaEventDestroy(begin);
}

int main()
{
  measure_cub(Input<std::uint64_t, std::uint32_t>(
    gen_uniform_buffer_sizes<std::uint32_t>(1024, 1024 * 1024)));
  return 0;
}
