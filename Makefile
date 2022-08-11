TARGET := fftconv_test

BUILD_DIR := ./build
SRC_DIRS := ./ ./src ./src_pocketfft
SRCS := test.cpp src/fftconv.cpp pocketfft.c
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
LDFLAGS := -lfftw3 -L/opt/homebrew/lib -larmadillo -lpthread

# Every folder in ./src will need to be passed to GCC so that it can find header files
INC_DIRS := ./ ./src ./src_pocketfft
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
INC_FLAGS := $(addprefix -I,$(INC_DIRS)) 

# The -MMD and -MP flags together generate Makefiles for us!
# These files will have .d instead of .o as the output.
CPPFLAGS := $(INC_FLAGS) -MMD -MP -Wall -Wno-sign-compare -I/opt/homebrew/include -Ofast
CXXFLAGS := -std=c++17

# The final build step.
$(BUILD_DIR)/$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# Build step for C source
$(BUILD_DIR)/%.c.o: %.c
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: test
test: $(BUILD_DIR)/$(TARGET)
	$(BUILD_DIR)/$(TARGET)

.PHONY: python
python:
	python3 setup.py build_ext --inplace

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -f *.so
