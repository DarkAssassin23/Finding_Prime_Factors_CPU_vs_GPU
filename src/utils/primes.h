#pragma once
#include <vector>
#include <cstdint>

void genPrimes(int minRange, int maxRange, std::vector<int> *primeList);
std::vector<int> multithreadPrimeGen(int maxNumber);
uint64_t getRandomProduct(std::vector<uint64_t> *primes);

std::vector<uint64_t> genPrimesGPU(void);