#pragma once
#include <vector>

void genPrimes(int minRange, int maxRange, std::vector<int> *primeList);
std::vector<int> multithreadPrimeGen(int maxNumber);
uint64_t getRandomProduct(std::vector<int> *primes);