#include <iostream>
#include <cstdint>
#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <math.h>
#include <thread>
#include <random>
#include <vector>
#include "primes.h"

bool isPrime(int number)
{
    if(number < 2) 
        return false;
    if(number == 2) 
        return true;
    if(number % 2 == 0) 
        return false;
    for (int x = 3; (x*x) < number; x+=2)
        if(number % x == 0) 
            return false;
    
    return true;
}

void genPrimes(int minRange, int maxRange, std::vector<int> *primeList)
{
    for(int x = minRange; x < maxRange; x++)
        if(isPrime(x))
            primeList->push_back(x);
}

std::vector<int> multithreadPrimeGen(int maxNumber)
{
    const auto processor_count = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::vector<std::vector<int>> totalPrimesListSegment(processor_count);
    int range = maxNumber / processor_count;

    for(int x=0;x<processor_count;x++)
    {
        int start = x * range;
        int end = (x+1) * range;
        threads.push_back(std::thread(genPrimes, start, end, &totalPrimesListSegment.at(x)));
    }

    for(int x=0;x<processor_count;x++)
        threads.at(x).join();

    std::vector<int> primeList;
    for(auto x : totalPrimesListSegment)
        primeList.insert(primeList.end(), x.begin(), x.end());

    return primeList;
}

int genRandomNum(int lower, int upper)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> genRandom(lower, upper);
    
    return genRandom(rng);
}

uint64_t getRandomProduct(std::vector<uint64_t> *primes)
{
    int p1 = genRandomNum(0, primes->size());
    int p2 = p1;
    do
    {
        p2 = genRandomNum(0, primes->size());
    }while (p2 == p1);

    return (static_cast<uint64_t>(primes->at(p1)) * static_cast<uint64_t>(primes->at(p2)));
}

// Barring minor changes, the follow code below is courtesy of the
// following stackoverlow post: https://stackoverflow.com/q/15622196
#define MAX_BLOCKS 256
#define THREADS_PER_BLOCK 256 //Must be a power of 2
#define BLOCK_SPACE 2 * THREADS_PER_BLOCK
#define MIN_PRIMES 10000000 // Minimum primes to generate

__global__ void initialize(uint64_t* isPrime, uint64_t n) 
{
    uint64_t idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    uint64_t step = gridDim.x * THREADS_PER_BLOCK;
    uint64_t i;
    for (i = idx; i <= 1; i += step)
        isPrime[i] = 0;

    for (; i < n; i += step)
        isPrime[i] = 1;
}

__global__ void clearMultiples(uint64_t* isPrime, uint64_t* primeList, 
                               uint64_t startInd, uint64_t endInd, 
                               uint64_t n) 
{
    uint64_t yidx = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t xidx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t ystep = gridDim.y * blockDim.y;
    uint64_t xstep = gridDim.x * blockDim.x;
    for (uint64_t pnum = startInd + yidx; pnum < endInd; pnum += ystep) 
    {
        uint64_t p = primeList[pnum];
        uint64_t pstart = p * (p + xidx);
        uint64_t pstep = p * xstep;
        for (uint64_t i = pstart; i < n; i += pstep)
            isPrime[i] = 0;

    }
}

__device__ void makeCounts(uint64_t* isPrime, uint64_t* addend, 
                           uint64_t start, uint64_t stop) 
{
    __shared__ uint64_t tmpCounts[BLOCK_SPACE];
    __shared__ uint64_t dumbCounts[BLOCK_SPACE];
    uint64_t idx = threadIdx.x;
    tmpCounts[idx] = ((start + idx) < stop) ? isPrime[start + idx] : 0;
    __syncthreads();
    uint64_t numEntries = THREADS_PER_BLOCK;
    uint64_t cstart = 0;
    while (numEntries > 1) 
    {
        uint64_t prevStart = cstart;
        cstart += numEntries;
        numEntries /= 2;
        if (idx < numEntries)
        {
            uint64_t i1 = idx * 2 + prevStart;
            tmpCounts[idx + cstart] = tmpCounts[i1] + tmpCounts[i1 + 1];
        }
        __syncthreads();
    }
    if (idx == 0)
    {
        dumbCounts[cstart] = tmpCounts[cstart];
        tmpCounts[cstart] = 0;
    }
    while (cstart > 0) 
    {
        uint64_t prevStart = cstart;
        cstart -= numEntries * 2;
        if (idx < numEntries) 
        {
            uint64_t v1 = tmpCounts[idx + prevStart];
            uint64_t i1 = idx * 2 + cstart;
            tmpCounts[i1 + 1] = tmpCounts[i1] + v1;
            tmpCounts[i1] = v1;
            dumbCounts[i1] = dumbCounts[i1 + 1] = dumbCounts[idx + prevStart];
        }
        numEntries *= 2;
        __syncthreads();
    }
    if (start + idx < stop) 
    {
        isPrime[start + idx] = tmpCounts[idx];
        addend[start + idx] = dumbCounts[idx];
    }
}

__global__ void createCounts(uint64_t* isPrime, uint64_t* addend, 
                             uint64_t lb, uint64_t ub) 
{
    uint64_t step = gridDim.x * THREADS_PER_BLOCK;
    for (uint64_t i = lb + blockIdx.x * THREADS_PER_BLOCK; i < ub; i += step) 
    {
        uint64_t start = i;
        uint64_t stop = min(i + step, ub);
        makeCounts(isPrime, addend, start, stop);
    }
}

__global__ void sumCounts(uint64_t* isPrime, uint64_t* addend, 
                          uint64_t lb, uint64_t ub, uint64_t* totalsum) 
{
    uint64_t idx = blockIdx.x;
    uint64_t s = 0;
    for (uint64_t i = lb + idx; i < ub; i += THREADS_PER_BLOCK) 
    {
        isPrime[i] += s;
        s += addend[i];
    }
    if (idx == 0)
        *totalsum = s;
}

__global__ void condensePrimes(uint64_t* isPrime, uint64_t* primeList, 
                               uint64_t lb, uint64_t ub,
                               uint64_t primeStartInd, uint64_t primeCount) 
{
    uint64_t idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    uint64_t step = gridDim.x * THREADS_PER_BLOCK;
    for (uint64_t i = lb + idx; i < ub; i += step)
    {
        uint64_t term = isPrime[i];
        uint64_t nextTerm = i + 1 == ub ? primeCount : isPrime[i + 1];
        if (term < nextTerm)
            primeList[primeStartInd + term] = i;
    }
}

std::vector<uint64_t> genPrimesGPU(void)
{
    // Get device properties of GPU 0 to get the amount of memory available
    // and calculate how many primes we can generate
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Half the memory for `isPrime` and the other half for `addend`
    // plus buffer to not allocate more VRAM than the card has
    const uint64_t MAX_RANGE = (prop.totalGlobalMem * 0.45f) 
                                / sizeof(uint64_t);
    const uint64_t n = max((uint64_t)MIN_PRIMES, MAX_RANGE);
    std::cout<<"Generating prime numbers from 0 - "<<n<<"\n";
    
    // If the GPU doesn't have enough memory to generate the minimum
    // number of primes, fall back to generating them with the CPU
    if(MAX_RANGE < MIN_PRIMES)
    {
        std::vector<int> tmp = multithreadPrimeGen((int) n);
        std::vector<uint64_t> result;
        std::copy(tmp.begin(), tmp.end(), 
                  back_inserter(result));

        return result;
    }

    uint64_t *isPrime, *addend, *numPrimes, *primeList;
    cudaError_t t = cudaMalloc((void**) &isPrime, n * sizeof(uint64_t));
    assert(t == cudaSuccess);
    
    t = cudaMalloc(&addend, n * sizeof(uint64_t));
    assert(t == cudaSuccess);
    
    t = cudaMalloc(&numPrimes, sizeof(uint64_t));
    assert(t == cudaSuccess);
    
    uint64_t primeBound = 2 * n / log(n);
    t = cudaMalloc(&primeList, primeBound * sizeof(uint64_t));
    assert(t == cudaSuccess);
    
    uint64_t numBlocks = min((uint64_t)MAX_BLOCKS,
                             (n + THREADS_PER_BLOCK - 1) 
                             / THREADS_PER_BLOCK);
    initialize<<<numBlocks, THREADS_PER_BLOCK>>>(isPrime, n);
    t = cudaDeviceSynchronize();
    assert(t == cudaSuccess);

    uint64_t bound = (uint64_t) ceil(sqrt(n));
    uint64_t lb;
    uint64_t ub = 2;
    uint64_t primeStartInd = 0;
    uint64_t primeEndInd = 0;

    while (ub < n) 
    {
        if (primeEndInd > primeStartInd) 
        {
            uint64_t lowprime;
            t = cudaMemcpy(&lowprime, primeList + primeStartInd, 
                           sizeof(uint64_t), cudaMemcpyDeviceToHost);
            assert(t == cudaSuccess);

            uint64_t numcols = n / lowprime;
            uint64_t numrows = primeEndInd - primeStartInd;
            uint64_t threadx = min(numcols, (uint64_t)THREADS_PER_BLOCK);
            uint64_t thready = min(numrows, THREADS_PER_BLOCK / threadx);
            uint64_t blockx = min(numcols / threadx, (uint64_t) MAX_BLOCKS);
            uint64_t blocky = min(numrows / thready, MAX_BLOCKS / blockx);

            dim3 gridsize(blockx, blocky);
            dim3 blocksize(threadx, thready);
            clearMultiples<<<gridsize, blocksize>>>(isPrime, primeList,
                                                    primeStartInd, 
                                                    primeEndInd, n);
            t = cudaDeviceSynchronize();
            assert(t == cudaSuccess);
        }
        lb = ub;
        ub *= 2;
        if (lb >= bound)
            ub = n;
        
        numBlocks = min((uint64_t)MAX_BLOCKS,
                        (ub - lb + THREADS_PER_BLOCK - 1) 
                        / THREADS_PER_BLOCK);

        createCounts<<<numBlocks, THREADS_PER_BLOCK>>>(isPrime, addend, 
                                                       lb, ub);
        t = cudaDeviceSynchronize();
        assert(t == cudaSuccess);

        sumCounts<<<THREADS_PER_BLOCK, 1>>>(isPrime, addend, lb, ub, 
                                            numPrimes);
        t = cudaDeviceSynchronize();
        assert(t == cudaSuccess);

        uint64_t primeCount;
        t = cudaMemcpy(&primeCount, numPrimes, sizeof(uint64_t),
                       cudaMemcpyDeviceToHost);
        assert(t == cudaSuccess);
        assert(primeCount > 0);

        primeStartInd = primeEndInd;
        primeEndInd += primeCount;
        condensePrimes<<<numBlocks, THREADS_PER_BLOCK>>>(isPrime, primeList,
                                                         lb, ub, 
                                                         primeStartInd, 
                                                         primeCount);
        t = cudaDeviceSynchronize();
        assert(t == cudaSuccess);
    }

    uint64_t *finalprimes = (uint64_t *) malloc(primeEndInd 
                                                * sizeof(uint64_t));
    t = cudaMemcpy(finalprimes, primeList, primeEndInd * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
    assert(t == cudaSuccess);

    t = cudaFree(isPrime);
    assert(t == cudaSuccess);

    t = cudaFree(addend);
    assert(t == cudaSuccess);

    t = cudaFree(numPrimes);
    assert(t == cudaSuccess);

    t = cudaFree(primeList);
    assert(t == cudaSuccess);
    
    std::vector<uint64_t> result;
    std::copy(&finalprimes[0], &finalprimes[primeEndInd], 
              back_inserter(result));

    free(finalprimes);

    return result;
}
