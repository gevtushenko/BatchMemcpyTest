#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/shuffle.h>

#include "cub/device/device_partition.cuh"
#include "cub/iterator/cache_modified_input_iterator.cuh"
#include "cub/iterator/cache_modified_output_iterator.cuh"
#include "cub/device/device_batch_memcpy.cuh"
#include "cub/block/block_load.cuh"
#include "cub/block/block_store.cuh"
#include "cub/warp/warp_load.cuh"
#include "cub/warp/warp_store.cuh"


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

template <typename T>
thrust::host_vector<T> gen_shuffled_buffer_sizes(std::size_t small_buffers,
                                                 std::size_t medium_buffers,
                                                 std::size_t large_buffers,
                                                 std::size_t small_buffer_size,
                                                 std::size_t medium_buffer_size,
                                                 std::size_t large_buffer_size)
{
  const std::size_t total_buffers = small_buffers + medium_buffers + large_buffers;
  thrust::host_vector<T> sizes(total_buffers);

  thrust::fill_n(sizes.begin(), small_buffers, small_buffer_size);
  thrust::fill_n(sizes.begin() + small_buffers, medium_buffers, medium_buffer_size);
  thrust::fill_n(sizes.begin() + small_buffers + medium_buffers, large_buffers, large_buffer_size);

  thrust::default_random_engine re;
  thrust::shuffle(sizes.begin(), sizes.end(), re);

  return sizes;
}

template <typename DataT, typename OffsetT>
class Input
{
  const std::size_t buffers {};
  std::size_t total_input_size {};

  mutable thrust::device_vector<DataT> input;
  mutable thrust::device_vector<DataT> output;

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

  void fill_input(DataT value) const
  {
    thrust::fill(input.begin(), input.end(), value);
  }

  void fill_output(DataT value) const
  {
    thrust::fill(output.begin(), output.end(), value);
  }

  void compare() const
  {
    if (output != input)
    {
      throw std::runtime_error("Wrong result!");
    }
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

  void* get_input_raw() const
  {
    return const_cast<DataT*>(thrust::raw_pointer_cast(input.data()));
  }

  void* get_output_raw() const
  {
    return const_cast<DataT*>(thrust::raw_pointer_cast(output.data()));
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
    return static_cast<float>(bytes) / 1024.0f / 1024.0f / 1024.0f;
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

  input.fill_input(DataT{42});
  input.fill_output(DataT{1});

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

  input.compare();

  report_result(ms, input);

  cudaEventDestroy(end);
  cudaEventDestroy(begin);
}


template <int BlockThreads,
          typename OffsetT>
__launch_bounds__(BlockThreads)
__global__ void naive_kernel(void **in_pointers,
                             void **out_pointers,
                             const OffsetT *sizes)
{
  using underlying_type = std::uint32_t;

  constexpr int items_per_thread = 4;
  constexpr int tile_size = items_per_thread * BlockThreads;

  using BlockLoadT =
    cub::BlockLoad<underlying_type,
                   BlockThreads,
                   items_per_thread,
                   cub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE>;

  using BlockStoreT =
    cub::BlockStore<underlying_type,
                    BlockThreads,
                    items_per_thread,
                    cub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE>;

  __shared__ union
  {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;

  } storage;

  const int buffer_id = blockIdx.x;
  auto in = reinterpret_cast<underlying_type*>(in_pointers[buffer_id]);
  auto out = reinterpret_cast<underlying_type*>(out_pointers[buffer_id]);
  const auto size = sizes[buffer_id];
  const auto size_in_elements = size / sizeof(underlying_type);
  const auto tiles = size_in_elements / tile_size;

  for (std::size_t tile = 0; tile < tiles; tile++)
  {
    cub::CacheModifiedInputIterator<cub::CacheLoadModifier::LOAD_CS, underlying_type> in_iterator(in);
    cub::CacheModifiedOutputIterator<cub::CacheStoreModifier::STORE_CS, underlying_type> out_iterator(out);

    underlying_type thread_data[items_per_thread];
    BlockLoadT(storage.load).Load(in_iterator, thread_data);
    BlockStoreT(storage.store).Store(out_iterator, thread_data);

    in += tile_size;
    out += tile_size;
  }
}


template <typename DataT,
          typename OffsetT>
void measure_naive(const Input<DataT, OffsetT> &input)
{
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  input.fill_input(DataT{24});
  input.fill_output(DataT{1});

  cudaEventRecord(begin);

  constexpr int block_threads = 256;
  naive_kernel<block_threads, OffsetT>
    <<<input.get_num_buffers(), block_threads>>>(
      input.get_input(),
      input.get_output(),
      input.get_buffer_sizes());

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms {};
  cudaEventElapsedTime(&ms, begin, end);

  input.compare();

  report_result(ms, input);

  cudaEventDestroy(end);
  cudaEventDestroy(begin);
}


template <int BlockThreads,
          typename OffsetT>
__launch_bounds__(BlockThreads)
__global__ void large_kernel(
    int large_buffers,
    const int *large_buffers_reordering,
    int *tiles_copied_ptr,

    void **in_pointers,
    void **out_pointers,
    const OffsetT *sizes)
{
  using underlying_type = std::uint32_t;

  constexpr int items_per_thread = 4;
  constexpr int tile_size = items_per_thread * BlockThreads;
  constexpr int tiles_per_request = 2;

  using BlockLoadT =
    cub::BlockLoad<underlying_type,
                   BlockThreads,
                   items_per_thread,
                   cub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE>;

  using BlockStoreT =
    cub::BlockStore<underlying_type,
                    BlockThreads,
                    items_per_thread,
                    cub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE>;

  __shared__ union
  {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;

  } storage;

  for (unsigned int bid = blockIdx.x % large_buffers; bid < large_buffers; bid += gridDim.x)
  {
    const int buffer_id = large_buffers_reordering[bid];

    auto in_origin =
      reinterpret_cast<underlying_type *>(in_pointers[buffer_id]);
    auto out_origin =
      reinterpret_cast<underlying_type *>(out_pointers[buffer_id]);
    const auto size             = sizes[buffer_id];
    const auto size_in_elements = size / sizeof(underlying_type);
    const auto tiles            = size_in_elements / tile_size;

    __shared__ int tiles_copied_cache;

    if (threadIdx.x == 0)
    {
      tiles_copied_cache = atomicAdd(tiles_copied_ptr + buffer_id,
                                     tiles_per_request);
    }
    __syncthreads();
    int tiles_copied = tiles_copied_cache;

    while (tiles_copied < tiles)
    {
      for (std::size_t tile = 0; tile < tiles_per_request; tile++)
      {
        if (tile + tiles_copied >= tiles)
        {
          break;
        }

        const OffsetT tile_offset = (tile + tiles_copied) * tile_size;
        const auto in             = in_origin + tile_offset;
        const auto out            = out_origin + tile_offset;

        cub::CacheModifiedInputIterator<cub::CacheLoadModifier::LOAD_CS,
                                        underlying_type>
          in_iterator(in);
        cub::CacheModifiedOutputIterator<cub::CacheStoreModifier::STORE_CS,
                                         underlying_type>
          out_iterator(out);

        underlying_type thread_data[items_per_thread];
        BlockLoadT(storage.load).Load(in_iterator, thread_data);
        BlockStoreT(storage.store).Store(out_iterator, thread_data);
      }

      if (threadIdx.x == 0)
      {
        tiles_copied_cache = atomicAdd(tiles_copied_ptr + buffer_id,
                                       tiles_per_request);
      }
      __syncthreads();
      tiles_copied = tiles_copied_cache;
    }
  }
}


template <typename DataT,
  typename OffsetT>
void measure_large(const Input<DataT, OffsetT> &input)
{
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);


  thrust::device_vector<int> buffers_reordering(input.get_num_buffers());
  thrust::sequence(buffers_reordering.begin(), buffers_reordering.end());
  const int *d_buffers_reordering = thrust::raw_pointer_cast(buffers_reordering.data());

  thrust::device_vector<int> tiles_copied(input.get_num_buffers());
  int *d_tiles_copied = thrust::raw_pointer_cast(tiles_copied.data());

  input.fill_input(DataT{24});
  input.fill_output(DataT{1});


  constexpr int block_threads = 256;


  int sm_count;
  int dev_id = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);

  // Get SM occupancy for the batch memcpy block-level buffers kernel
  int max_occupancy;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_occupancy,
    large_kernel<block_threads, OffsetT>,
    block_threads,
    0);

  const int grid_size = max_occupancy * sm_count;


  cudaEventRecord(begin);

  large_kernel<block_threads, OffsetT>
    <<<grid_size, block_threads>>>(

      input.get_num_buffers(),
      d_buffers_reordering,
      d_tiles_copied,

      input.get_input(),
      input.get_output(),
      input.get_buffer_sizes());

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms{};
  cudaEventElapsedTime(&ms, begin, end);

  input.compare();

  report_result(ms, input);

  cudaEventDestroy(end);
  cudaEventDestroy(begin);
}


template <typename DataT,
          typename OffsetT>
void measure_memcpy(const Input<DataT, OffsetT> &input)
{
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);


  input.fill_input(DataT{24});
  input.fill_output(DataT{1});

  cudaEventRecord(begin);

  cudaMemcpyAsync(input.get_output_raw(),
                  input.get_input_raw(),
                  input.get_bytes_written(),
                  cudaMemcpyDeviceToDevice);

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms{};
  cudaEventElapsedTime(&ms, begin, end);

  input.compare();

  report_result(ms, input);

  cudaEventDestroy(end);
  cudaEventDestroy(begin);
}


template <typename OffsetT>
struct LargeSegmentsSelectorT
{
  OffsetT value{};
  const OffsetT *d_sizes{};

  __host__ __device__ __forceinline__
  LargeSegmentsSelectorT(OffsetT value,
                         const OffsetT *d_sizes)
    : value(value)
    , d_sizes(d_sizes)
  {}

  __host__ __device__ __forceinline__ bool
  operator()(int segment_id) const
  {
    return d_sizes[segment_id] > value;
  }
};


template <typename OffsetT>
struct SmallSegmentsSelectorT
{
  OffsetT value{};
  const OffsetT *d_sizes{};

  __host__ __device__ __forceinline__
  SmallSegmentsSelectorT(OffsetT value,
                         const OffsetT *d_sizes)
    : value(value)
    , d_sizes(d_sizes)
  {}

  __host__ __device__ __forceinline__ bool
  operator()(unsigned int segment_id) const
  {
    return d_sizes[segment_id] < value;
  }
};

template <int BlockThreads,
          typename LargeBuffersReorderingT,
          typename MediumBuffersReorderingT,
          typename OffsetT>
__launch_bounds__(BlockThreads)
__global__ void partitioned_kernel(
  int num_buffers,
  const int *d_group_sizes,
  LargeBuffersReorderingT large_buffers_reordering,
  MediumBuffersReorderingT medium_buffers_reordering,
  int *small_buffers_reordering,
  int *tiles_copied_ptr,

  void **in_pointers,
  void **out_pointers,
  const OffsetT *sizes)
{
  using underlying_type = std::uint32_t;

  const unsigned int large_buffers = d_group_sizes[0];
  const unsigned int small_buffers = d_group_sizes[1];
  const unsigned int medium_buffers = num_buffers - large_buffers - small_buffers;

  constexpr int items_per_thread  = 4;
  constexpr int tile_size         = items_per_thread * BlockThreads;
  constexpr int tiles_per_request = 2;

  using BlockLoadT =
    cub::BlockLoad<underlying_type,
                   BlockThreads,
                   items_per_thread,
                   cub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE>;

  using BlockStoreT =
    cub::BlockStore<underlying_type,
                    BlockThreads,
                    items_per_thread,
                    cub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE>;

  constexpr int warp_size = 4;

  /*
   *
template <typename          InputT,
          int               ITEMS_PER_THREAD,
          WarpLoadAlgorithm ALGORITHM            = WARP_LOAD_DIRECT,
          int               LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS,
   */

  using WarpLoadT = cub::WarpLoad<underlying_type,
                                  items_per_thread,
                                  cub::WarpLoadAlgorithm::WARP_LOAD_VECTORIZE,
                                  warp_size>;
  using WarpStoreT =
    cub::WarpStore<underlying_type,
                   items_per_thread,
                   cub::WarpStoreAlgorithm::WARP_STORE_VECTORIZE,
                   warp_size>;

  __shared__ union
  {
    typename BlockLoadT::TempStorage block_load;
    typename BlockStoreT::TempStorage block_store;

    typename WarpLoadT::TempStorage warp_load;
    typename WarpStoreT::TempStorage warp_store;
  } storage;


  for (unsigned int bid = blockIdx.x; bid < medium_buffers; bid += gridDim.x)
  {
    const int buffer_id = medium_buffers_reordering[bid];

    auto in = reinterpret_cast<underlying_type *>(in_pointers[buffer_id]);
    auto out = reinterpret_cast<underlying_type *>(out_pointers[buffer_id]);
    const auto size             = sizes[buffer_id];
    const auto size_in_elements = size / sizeof(underlying_type);
    const auto tiles            = size_in_elements / tile_size;

    for (std::size_t tile = 0; tile < tiles; tile++)
    {
      cub::CacheModifiedInputIterator<cub::CacheLoadModifier::LOAD_CS, underlying_type> in_iterator(in);
      cub::CacheModifiedOutputIterator<cub::CacheStoreModifier::STORE_CS, underlying_type> out_iterator(out);

      underlying_type thread_data[items_per_thread];
      BlockLoadT(storage.block_load).Load(in_iterator, thread_data);
      BlockStoreT(storage.block_store).Store(out_iterator, thread_data);

      in += tile_size;
      out += tile_size;
    }
  }

  if (large_buffers > 0)
  {
    for (unsigned int bid = blockIdx.x % large_buffers; bid < large_buffers; bid += gridDim.x)
    {
      const int buffer_id = large_buffers_reordering[bid];

      auto in_origin = reinterpret_cast<underlying_type *>(in_pointers[buffer_id]);
      auto out_origin = reinterpret_cast<underlying_type *>(out_pointers[buffer_id]);
      const auto size             = sizes[buffer_id];
      const auto size_in_elements = size / sizeof(underlying_type);
      const auto tiles            = (size_in_elements + tile_size - 1) / tile_size;

      __shared__ int tiles_copied_cache;

      if (threadIdx.x == 0)
      {
        tiles_copied_cache = atomicAdd(tiles_copied_ptr + buffer_id,
                                       tiles_per_request);
      }
      __syncthreads();
      int tiles_copied = tiles_copied_cache;

      bool process_last_tile = false;

      while (tiles_copied < tiles)
      {
        for (std::size_t tile = 0; tile < tiles_per_request; tile++)
        {
          if (tile + tiles_copied == tiles - 1)
          {
            process_last_tile = true;
            break;
          }

          if (tile + tiles_copied >= tiles)
          {
            break;
          }

          const OffsetT tile_offset = (tile + tiles_copied) * tile_size;
          const auto in             = in_origin + tile_offset;
          const auto out            = out_origin + tile_offset;

          cub::CacheModifiedInputIterator<cub::CacheLoadModifier::LOAD_CS, underlying_type> in_iterator(in);
          cub::CacheModifiedOutputIterator<cub::CacheStoreModifier::STORE_CS, underlying_type> out_iterator(out);

          underlying_type thread_data[items_per_thread];
          BlockLoadT(storage.block_load).Load(in_iterator, thread_data);
          BlockStoreT(storage.block_store).Store(out_iterator, thread_data);
        }

        if (process_last_tile)
        {
          const OffsetT tile_offset = (tiles - 1) * tile_size;
          const auto in = in_origin + tile_offset;
          const auto out = out_origin + tile_offset;

          const int valid_items = size_in_elements - (tiles - 1) * tile_size;

          cub::CacheModifiedInputIterator<cub::CacheLoadModifier::LOAD_CS, underlying_type> in_iterator(in);
          cub::CacheModifiedOutputIterator<cub::CacheStoreModifier::STORE_CS, underlying_type> out_iterator(out);

          underlying_type thread_data[items_per_thread];
          BlockLoadT(storage.block_load).Load(in_iterator, thread_data, valid_items);
          BlockStoreT(storage.block_store).Store(out_iterator, thread_data, valid_items);
          break;
        }
        else
        {
          if (threadIdx.x == 0)
          {
            tiles_copied_cache = atomicAdd(tiles_copied_ptr + buffer_id,
                                           tiles_per_request);
          }
          __syncthreads();
          tiles_copied = tiles_copied_cache;
        }
      }
    }
  }

  constexpr unsigned int warp_tile_size = warp_size * items_per_thread;
  const unsigned int warp_id = (BlockThreads * blockIdx.x + threadIdx.x) / warp_size;
  const unsigned int total_warps = (BlockThreads * gridDim.x) / warp_size;

  for (unsigned int bid = warp_id; bid < small_buffers; bid += total_warps)
  {
    const int buffer_id = small_buffers_reordering[bid];

    auto in = reinterpret_cast<underlying_type *>(in_pointers[buffer_id]);
    auto out = reinterpret_cast<underlying_type *>(out_pointers[buffer_id]);
    const auto size             = sizes[buffer_id];
    const auto size_in_elements = size / sizeof(underlying_type);
    const auto tiles            = size_in_elements / warp_tile_size;

    for (std::size_t tile = 0; tile < tiles; tile++)
    {
      cub::CacheModifiedInputIterator<cub::CacheLoadModifier::LOAD_CS, underlying_type> in_iterator(in);
      cub::CacheModifiedOutputIterator<cub::CacheStoreModifier::STORE_CS, underlying_type> out_iterator(out);

      underlying_type thread_data[items_per_thread];
      WarpLoadT(storage.warp_load).Load(in_iterator, thread_data);
      WarpStoreT(storage.warp_store).Store(out_iterator, thread_data);

      in += warp_tile_size;
      out += warp_tile_size;
    }
  }
}


template <typename DataT,
          typename OffsetT>
void measure_partition(const Input<DataT, OffsetT> &input)
{
  const std::size_t num_buffers = input.get_num_buffers();

  thrust::counting_iterator<int> buffer_ids(0);

  thrust::device_vector<int> small_and_large(num_buffers);
  thrust::device_vector<int> queue_and_medium(num_buffers);

  int *d_small_and_large = thrust::raw_pointer_cast(small_and_large.data());
  int *d_queue_and_medium = thrust::raw_pointer_cast(queue_and_medium.data());

  auto medium = thrust::make_reverse_iterator(d_queue_and_medium + num_buffers);
  auto large = thrust::make_reverse_iterator(d_small_and_large + num_buffers);

  const OffsetT *d_sizes = input.get_buffer_sizes();

  const OffsetT small_segment_max_size = 1024; // bytes
  SmallSegmentsSelectorT<OffsetT> small_selector(small_segment_max_size, d_sizes);

  const OffsetT large_segment_min_size = 1024 * 1024; // bytes
  LargeSegmentsSelectorT<OffsetT> large_selector(large_segment_min_size, d_sizes);

  thrust::device_vector<int> group_sizes(2);
  int *d_group_sizes = thrust::raw_pointer_cast(group_sizes.data());


  std::size_t temp_storage_bytes;
  cub::DevicePartition::If(nullptr,
                           temp_storage_bytes,
                           buffer_ids,
                           large,
                           d_small_and_large,
                           medium,
                           d_group_sizes,
                           num_buffers,
                           small_selector,
                           large_selector);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  std::uint8_t *d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());


  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);


  input.fill_input(DataT{35});
  input.fill_output(DataT{2});

  cudaEventRecord(begin);

  constexpr int block_threads = 256;

  int dev_id = 0;
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);

  // Get SM occupancy for the batch memcpy block-level buffers kernel
  int max_occupancy;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_occupancy,
    partitioned_kernel<block_threads, decltype(large), decltype(medium), OffsetT>,
    block_threads,
    0);

  const int grid_size = max_occupancy * sm_count;

  cudaMemsetAsync(d_queue_and_medium, 0, sizeof(int) * num_buffers);

  cub::DevicePartition::If(d_temp_storage,
                           temp_storage_bytes,
                           buffer_ids,
                           large,
                           d_small_and_large,
                           medium,
                           d_group_sizes,
                           num_buffers,
                           large_selector,
                           small_selector);

  partitioned_kernel<block_threads>
    <<<grid_size, block_threads>>>(num_buffers,
                                   d_group_sizes,
                                   large,
                                   medium,
                                   d_small_and_large,
                                   d_queue_and_medium,

                                   input.get_input(),
                                   input.get_output(),
                                   input.get_buffer_sizes());

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms{};
  cudaEventElapsedTime(&ms, begin, end);

  input.compare();

  report_result(ms, input);

  cudaEventDestroy(end);
  cudaEventDestroy(begin);
}


int main()
{
  const auto input = Input<std::uint32_t, std::uint32_t>(
    gen_shuffled_buffer_sizes<std::uint32_t>(
      0, // small
      0, // medium
      300, // large,
      96,
      256 * 4 * 16,
      2 * 1024 * 1024));

  // 1024 * 1024 buffers of 256 elements => 46%
  // 1024 buffers of 1024 * 1024 elements => 78%
  // 2 buffers of 256 * 1024 * 1024 elements => 79%
  measure_cub(input);
  // measure_naive(input);
  // measure_large(input);
  measure_memcpy(input);

  measure_partition(input);

  return 0;
}
