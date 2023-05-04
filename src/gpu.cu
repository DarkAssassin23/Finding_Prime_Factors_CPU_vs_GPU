#include <algorithm>
#include <stdint.h>
#include "gpu.h"

// Given a range to check, numsToCheck, start at the top of the range
// and work backwards. If a factor is found, any potential factor less 
// than it is irrelevant and threads working in a range less than
// the prime found can be stopped.
__global__ void searchPrimesGPU(const uint64_t numsToCheck, uint64_t product, uint64_t *maxPrime, uint64_t *killLessThan)
{
    uint64_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start = numsToCheck - 1 - global_id; 

    while(start >= 3 && start < numsToCheck && *killLessThan < start)
    {
        uint64_t val = (start*2)+3;

        if(product%val == 0 && val > *maxPrime)
        {
            *maxPrime = val;
            *killLessThan = start;
            return;
        }

        global_id += blockDim.x * gridDim.x;
        start = numsToCheck - 1 - global_id;
    }
}

__host__ std::vector<uint64_t> searchPrimesGPULauncher(const uint64_t N)
{
    std::vector<uint64_t> out = {2};
    uint64_t maxRange = sqrt(N)/2 + 1;
    uint64_t maxPrime = 0, *maxPrime_d;
    uint64_t killLessThan = 0, *killLessThan_d;

    // Get device properties of GPU 0 to calculate
    // max threads and blocks
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    cudaMalloc((void**) &maxPrime_d, sizeof(uint64_t));
    cudaMalloc((void**) &killLessThan_d, sizeof(int64_t));
    cudaMemcpy(maxPrime_d, &killLessThan, sizeof(uint64_t), cudaMemcpyHostToDevice);

    const uint64_t MAX_THREADS = prop.maxThreadsPerBlock;
    const uint64_t MAX_BLOCKS = prop.maxBlocksPerMultiProcessor * prop.multiProcessorCount;;
    
    // In the event our number is small, don't allocate more
    // resources than needed
    int threads = std::min(maxRange, static_cast<uint64_t>(MAX_THREADS));
    int blocks = std::min(MAX_BLOCKS, static_cast<uint64_t>((maxRange+MAX_THREADS-1)/MAX_THREADS));

    searchPrimesGPU<<<blocks,threads>>>(maxRange, N, maxPrime_d, killLessThan_d);

    cudaMemcpy(&maxPrime, maxPrime_d, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(maxPrime_d);

    out.push_back(maxPrime);
    return out;
}

__host__ void initGPU()
{
    cudaFree(0);
}
