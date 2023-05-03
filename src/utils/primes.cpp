#include <thread>
#include <random>
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

uint64_t getRandomProduct(std::vector<int> *primes)
{
    int p1 = genRandomNum(0, primes->size());
    int p2 = p1;
    do
    {
        p2 = genRandomNum(0, primes->size());
    }while (p2 == p1);

    return (static_cast<uint64_t>(primes->at(p1)) * static_cast<uint64_t>(primes->at(p2)));
}