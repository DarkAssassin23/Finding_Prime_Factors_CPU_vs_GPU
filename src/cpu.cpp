#include <thread>
#include <stdint.h>
#include <math.h>

#include "cpu.h"

std::vector<uint64_t> searchPrimesCPUSingleThreaded(const uint64_t N)
{
    std::vector<uint64_t> out = {2};
    uint64_t maxPrime = out.at(0);
    uint64_t end = sqrt(N);
    if(end % 2 == 0)
        end++;
    for(uint64_t i=end;i>1;i-=2)
    {
        // Check if i is a prime and factor of N
        if(N%i == 0)
        {
            maxPrime = i;
            break;
        }
    }
    
    if(maxPrime != out.at(0))
        out.push_back(maxPrime);

    return out;
}

void searchPrimesCPUMultithreaded(const uint64_t N, uint64_t start, const uint64_t end, std::vector<uint64_t>* primes, uint64_t *maxPrime, int threadID)
{
    // Make sure our starting point is odd
    // since we only care about odd numbers
    if(start%2==0)
        start++;

    for(uint64_t i=start; i>end && killThreadsLessThan < threadID;i-=2)
    {
        // Check if i is a prime and factor of N
        if(N%i == 0 && i > *maxPrime)
        {
            // Make sure we don't have a race condition
            primesMutex.lock();
            if(i > *maxPrime)
            {
                *maxPrime = i;
                killThreadsLessThan = threadID;
                primesMutex.unlock();
                return;
            }
            primesMutex.unlock();
        }
    }
}

std::vector<uint64_t> launchSearchPrimesCPUMultithreaded(const uint64_t N)
{
    std::vector<uint64_t> primes = {2};
    const auto processor_count = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    uint64_t range = std::ceil(sqrt(N)/processor_count);
    uint64_t maxPrime = 2;
    killThreadsLessThan = -1;

    for(int x=0;x<processor_count;x++)
    {
        // Since we are going backwards, start is higher
        uint64_t end = x * range;
        uint64_t start = (x+1) * range;
        threads.push_back(std::thread(searchPrimesCPUMultithreaded, N, start, end, &primes, &maxPrime, x));
    }

    for(int x=0;x<processor_count;x++)
        threads.at(x).join();

    primes.push_back(maxPrime);
    return primes;
}

std::vector<uint64_t> searchPrimesCPU(const uint64_t N, bool multithread)
{
    if(multithread)
        return launchSearchPrimesCPUMultithreaded(N);

    return searchPrimesCPUSingleThreaded(N);
}