#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <stdint.h>
#include <math.h>

#include "timer.h"
#include "cpu.h"
#include "gpu.h"
#include "primes.h"
#include "utils.h"

// Point at witch multithreading actually improves
// performance, based on testing
// NOTE: Testing done on 8 threads from an E5-2695 v2 CPU,
// different thread counts and different CPUs will most likely
// yield different results, and this trigger should be 
// updated accordingly for optimal performance
const uint64_t MULTITHREADED_SEARCH_TRIGGER = (UINT32_MAX * 4);

using std::cout;
using std::vector;

bool verbose = false;
bool verboseMinimal = true;

void processAndDisplayCPU(uint64_t N)
{
    Timer t;
    double timeCPU = 0;
    bool multithread = false;

    vector<uint64_t> v;

    multithread = (N>MULTITHREADED_SEARCH_TRIGGER);
    if(multithread && verbose)
        cout<<"Multithreaded search on\n";

    t.start();
    // If N is divisable by 0, or a perfect square there's no need to
    // find the prime factors
    if(N%2!=0 && !isPerfectSquare(N))
        v = searchPrimesCPU(N, multithread);
    else if(N%2==0)
    {
        v.push_back(2);
        if(verbose)
            cout << "Search skipped since " << N << " is even\n";
    }
    else if(isPerfectSquare(N))
    {
        v.push_back(sqrt(N));
        if(verbose)
            cout << "Search skipped since " << N << " is a perfect square\n";
    }
    timeCPU = t.time_elapsed_milliseconds();

    if(verbose)
        cout<<"--> Search time: "<<timeCPU<<" ms\n";

    //Factorization of N
    t.start();
    vector<uint64_t> primes ={};
    calculateFactors(N,&v,&primes);
    timeCPU = t.time_elapsed_milliseconds();

    if(verbose)
        cout<<"--> Factorization time: "<<timeCPU<<" ms\n";

    if(verbose || verboseMinimal)
    {
        cout<<"Prime factors of "<<N<<": ";

        if(primes.size()==1)
            cout<<"1 ";
        for(uint64_t prime : primes)
            cout<<prime<<" ";

        cout<<"\n";
    }
}

void processAndDisplayGPU(uint64_t N)
{
    Timer t;
    double timeGPU = 0;
    vector<uint64_t> v;

    t.start();
    // If N is divisable by 0, or a perfect square there's no need to
    // find the prime factors
    if(N%2!=0 && !isPerfectSquare(N))
        v = searchPrimesGPULauncher(N);
    else if(N%2==0)
    {
        v.push_back(2);
        if(verbose)
            cout << "Search skipped since " << N << " is even\n";
    }
    else if(isPerfectSquare(N))
    {
        v.push_back(sqrt(N));
        if(verbose)
            cout << "Search skipped since " << N << " is a perfect square\n";
    }
    timeGPU = t.time_elapsed_milliseconds();

    if(verbose)
        cout<<"--> Search time: "<<timeGPU<<" ms\n";

    t.start();
    vector<uint64_t> primes ={};
    calculateFactors(N,&v,&primes);
    timeGPU = t.time_elapsed_milliseconds();

    if(verbose)
        cout<<"--> Factorization time: "<<timeGPU<<" ms\n";

    if(verbose || verboseMinimal)
    { 
        cout<<"Prime factors of "<<N<<": ";
        if(primes.size()==1)
            cout<<"1 ";
        for(uint64_t prime : primes)
            cout<<prime<<" ";

        cout<<"\n";
    }

}

void printUsage(const char *prog)
{
    cout << "Usage: "<<prog<<" -p [value] OR "<<prog<<" -f [filename] OR "<<prog<<" -r [num]\n";
    cout << " -p [value]\t\tCalculate prime factors of a single number\n";
    cout << " -f [filename]\t\tCalculate prime factors of all values in the file\n";
    cout << " -r [num]\t\tGenerate [num] random numbers and calculate their primes\n";
    cout << "\n";

    cout << "Optional Flags:\n";
    cout << "  -v\t\t\tVerbose mode\n";
    cout << "  -a\t\t\tFind average compute time\n";
    cout << "  -cpu=[true/false]\tWhether to run on CPU (default is true)\n";
    cout << "  -gpu=[true/false]\tWhether to run on GPU (default is true)\n";
    cout << "\n";

    cout << "Examples:\n";
    cout << "  "<<prog<<" -p 4158294289 -v -cpu=false -gpu=true\n";
    cout << "  "<<prog<<" -f values.txt\n";
}

int main(int argc, char const *argv[]) 
{
    if(argc < 3 )
    {
        printUsage(argv[0]);
        return 1;
    }

    bool usingFile = strcmp(argv[1],"-f")==0;
    vector<uint64_t> products;

    if(usingFile)
    {
        const char *file = argv[2];
        if(!getProducts(file, &products))
        {
            cout<<"Error reading file: "<<file<<"\n";
            return 1;
        }
    }
    else if(strcmp(argv[1],"-r")==0)
    {
        const int MAX_RANGE = 50000000;
        cout<<"Generating prime numbers from 0 - "<<MAX_RANGE<<"\n";
        vector<int> primes = multithreadPrimeGen(MAX_RANGE);
        cout<<"Done.\n";

        int nums = std::stoull(argv[2]);
        if(nums == 0)
            nums = 1;
        
        for(int x=0; x<nums; x++)
            products.push_back(getRandomProduct(&primes));
        
    }
    else
        products.push_back(std::stoull(argv[2]));

    bool averageCompute = false;
    bool useCPU = true;
    bool useGPU = true;
    if(argc >= 4)
    {
        for(int x=2;x<argc;x++)
        {
            std::string arg = argv[x];
            if(strcmp(argv[x],"-v")==0)
                verbose = true;
            else if(strcmp(argv[x],"-a")==0)
                averageCompute = true;
            else if(arg.substr(0,4).compare("-cpu")==0)
                useCPU = arg.compare("-cpu=true")==0;
            else if(arg.substr(0,4).compare("-gpu")==0)
                useGPU = arg.compare("-gpu=true")==0;
        }
    }

    int runsPerProduct = 1;
    if(averageCompute)
    {
        runsPerProduct = 7;
        verboseMinimal = false;
    }

    if(useCPU)
    {
        Timer t;
        t.start();
        
        for(uint64_t product : products)
        {  
            cout<<std::setfill('=')<<std::setw(getDigits(product)+41)<<"="<<"\n";
            cout<<" Finding prime factors of "<< product <<" using the CPU\n";
            cout<<std::setfill('=')<<std::setw(getDigits(product)+41)<<"="<<"\n";
            vector<double> runtimes(runsPerProduct);
            for(int x=0;x<runsPerProduct;x++)
            {
                Timer ti;
                ti.start();
                processAndDisplayCPU(product); 
                runtimes.at(x) = ti.time_elapsed_milliseconds();
                if(runsPerProduct > 1)
                    cout<< "Run " << (x+1) << " finished in "<<runtimes.at(x)<<" ms\n";
            }
            if(runsPerProduct > 1)
                cout<<"Average runtime: "<<getAverageRuntime(&runtimes)<<" ms\n";
        }
            
        cout<<"\nTotal CPU execution time: "<<t.time_elapsed_milliseconds()<<" ms\n";
    }
    cout<<"\n";
    if(useGPU)
    {
        Timer t;
        t.start();
        for(uint64_t product : products)
        {
            cout<<std::setfill('=')<<std::setw(getDigits(product)+41)<<"="<<"\n";
            cout<<" Finding prime factors of "<< product <<" using the GPU\n";
            cout<<std::setfill('=')<<std::setw(getDigits(product)+41)<<"="<<"\n";
            vector<double> runtimes(runsPerProduct);
            for(int x=0;x<runsPerProduct;x++)
            {
                Timer ti;
                ti.start();
                processAndDisplayGPU(product); 
                runtimes.at(x) = ti.time_elapsed_milliseconds();
                if(runsPerProduct > 1)
                    cout<< "Run " << (x+1) << " finished in "<<runtimes.at(x)<<" ms\n";
                
            }
            if(runsPerProduct > 1)
                cout<<"Average runtime: "<<getAverageRuntime(&runtimes)<<" ms\n";
        }
             
        cout<<"\nTotal GPU execution time: "<<t.time_elapsed_milliseconds()<<" ms\n";
    }

    return 0;
}
