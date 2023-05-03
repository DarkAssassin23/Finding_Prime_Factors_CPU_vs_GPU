#include "timer.h"

void Timer::start()
{ 
    epoch = Clock::now(); 
}
double Timer::time_elapsed_milliseconds() const
{ 
    return duration_cast<microseconds>(Clock::now() - epoch).count() / 1000.0;
}