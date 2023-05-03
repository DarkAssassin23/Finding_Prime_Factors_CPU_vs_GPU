#pragma once
#include <chrono>
using namespace std::chrono;

class Timer 
{
    public:
        typedef high_resolution_clock Clock;
        void start();
        double time_elapsed_milliseconds() const;
    private:
        Clock::time_point epoch;
};