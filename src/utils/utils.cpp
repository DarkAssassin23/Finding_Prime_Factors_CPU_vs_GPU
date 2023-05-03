#include <fstream>
#include <string>
#include <algorithm>
#include <stdint.h>
#include <math.h>

#include "utils.h"

bool isPerfectSquare(const uint64_t N)
{
    return (uint64_t (sqrt(N)) * uint64_t (sqrt(N)) == N);
}

int getDigits(uint64_t N)
{
    int digits = 0;
    for(digits = 0; N > 0; digits++)
        N = N / 10;
    return digits;
}

double getAverageRuntime(std::vector<double> *runtimes)
{
    if(runtimes->size() == 1)
        return runtimes->at(0);

    std::sort(runtimes->begin(), runtimes->end());
    double avgTime = 0;

    // Ignore the slowest and fastest time and take the average
    for(int x=1;x<runtimes->size()-1;x++)
        avgTime += runtimes->at(x);

    return (avgTime / (runtimes->size()-2));
}

void calculateFactors(const uint64_t N, std::vector<uint64_t>* possiblePrimes, std::vector<uint64_t>* primes)
{
    for (int x = possiblePrimes->size()-1; x >= 0 && N != 1; x--)
    {
        uint64_t val = possiblePrimes->at(x);
        if(N % val == 0)
        {
            primes->push_back(val);
            primes->push_back(N/val);
            return;
        }
    }
    if(primes->size()==0)
        primes->push_back(N);
}

bool getProducts(const char* file, std::vector<uint64_t>* products)
{
    std::string line;
    std::ifstream productFile(file);
    if (productFile.is_open())
    {
        while (std::getline(productFile,line))
            products->push_back(std::stoull(line));

        productFile.close();
        return true;
    }
    return false;
}