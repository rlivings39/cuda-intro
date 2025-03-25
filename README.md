# My initial experiments with CUDA

Resources

* https://developer.nvidia.com/blog/even-easier-introduction-cuda/
* https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/
* https://docs.nvidia.com/cuda/cuda-c-programming-guide/

## Notes

Unified memory in CUDA provides one memory space available to all CPUs and GPUs on the system. `cudaMallocManaged()` returns a poniter to such memory available from the host and device. Just use `cudaFree()` to free it.

Define a CUDA kernel by using `__global__` on a function. This signifies that it can be run on the device.

To launch a kernel, say `add`, on the device call it with `add<<<gridDim, blockDim>>>(N, x, y);`. This is called the **execution configuration**.

The CPU thread doesn't wait for kernels to finish. If you need to use the output of kernels on the host call `cudaDeviceSynchronize()`.

Kernels run in blocks of threads that are a multiple of 32. Setting `blockDim` specifies how many threads to use. In your kernel code, `blockDim.x` is the number of threads in the current block and `threadIdx.x` is the index of the current thread within its block.

CUDA GPUs have several parallel processors grouped into streaming multiprocessors (SMs). Each SM runs multiple concurrent thread blocks. If you have `N` elements to process and 256 threads per block then you can launch an appropriate number of blocks by just dividing and rounding up

```cpp
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;
add<<<numBlocks, blockSize>>>(N, x, y);
```
CUDA also provides `gridDim.x` which is the number of blocks and `blockIdx.x` which is the block index in the grid.

A common indexing idiom for 1D arrays is `blockIdx.x * blockDim.x + threadIdx.x`. So block 0 handles `[0, blockDim.x)`, block 1 handles `[blockDim.x, 2*blockDim.x)` and so on.

Then we stride by `blockDim.x * gridDim.x`. This is called a grid-stride loop

```cpp
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
```
