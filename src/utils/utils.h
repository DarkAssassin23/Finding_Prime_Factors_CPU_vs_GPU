#pragma once
#include <vector>

bool isPerfectSquare(const uint64_t N);
int getDigits(uint64_t N);
double getAverageRuntime(std::vector<double> *runtimes);
void calculateFactors(const uint64_t N, std::vector<uint64_t>* possiblePrimes, std::vector<uint64_t>* primes);
bool getProducts(const char* file, std::vector<uint64_t>* products);
