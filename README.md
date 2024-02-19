# Finding Prime Factors: CPU vs. GPU

## About
Find the prime factors of a number or list of numbers

---------------

## Requirements 
In order to build the project you must have the following:

* C++11 or newer
* NVIDIA GPU
* NVIDIA CUDA Compiler (nvcc) 
  * Make sure nvcc and the CUDA libraries are apart of your path

> [!NOTE]  This code has only has been tested on Linux (RHEL 9.1).
> It should work on Windows too, but has not been tested. If you plan to
> use Windows, all the above requirements apply, but the `makefile`
> won't work unless you have `make` installed through a tool like 
> <a href="https://www.msys2.org" target="new">MSYS2</a>. 
> If you are using Visual Studio, you should be able to import the source 
> code into a Visual Studio CUDA project and compile and run it that way.

---------------

## Building
To build the project, use the included `makefile` by typing:
```
make
```

---------------

## Usage
```
Usage: ./factor -p [value] OR ./factor -f [filename] OR ./factor -r [num]
 -p [value]         Calculate prime factors of a single number
 -f [filename]      Calculate prime factors of all values in the file
 -r [num]           Generate [num] random numbers and calculate their primes

Optional Flags:
  -v                Verbose mode
  -a                Find average compute time
  -cpu=[true/false] Whether to run on CPU (default is true)
  -gpu=[true/false] Whether to run on GPU (default is true)

Examples:
  ./factor -p 4158294289 -v -cpu=false -gpu=true
  ./factor -f values.txt
```
