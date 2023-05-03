TARGET = factor

CXX = g++ -std=c++11
NVCC = nvcc

NVCCFLAGS = -O3
CPPFLAGS = -O3

OBJDIR = obj

INCLUDES = -I src -I src/utils

CPPOBJ = $(patsubst %.cpp, $(OBJDIR)/%_cpp.o, $(shell find . -type f -name "*.cpp"))
GPUOBJ = $(patsubst %.cu, $(OBJDIR)/%_cu.o, $(shell find . -type f -name "*.cu"))
HEADERS = $(shell find . -type f -name "*.h")


default: $(TARGET)

$(TARGET): $(GPUOBJ) $(CPPOBJ) 
	@$(NVCC) $(CPPOBJ) $(GPUOBJ) -o $(TARGET) 
	@echo "--> Created:  " $(TARGET)

$(OBJDIR)/%_cpp.o: %.cpp $(HEADERS)
	@mkdir -p $(@D)
	@$(CXX) $(CPPFLAGS) $(INCLUDES) -c $< -o $@
	@echo "--> Compiled: " $<

$(OBJDIR)/%_cu.o: %.cu $(HEADERS)
	@mkdir -p $(@D)
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
	@echo "--> Compiled: " $<

clean:
	rm -rf $(OBJDIR) $(TARGET)
