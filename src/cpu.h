#pragma once
#include <vector>
#include <mutex>
#include <atomic>

static std::mutex primesMutex;
static std::atomic<int> killThreadsLessThan;

std::vector<uint64_t> searchPrimesCPU(const uint64_t N, bool multithread);
