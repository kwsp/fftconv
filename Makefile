TARGET := fftconv_test

BUILD_DIR := ./build
SRC_DIRS := ./
SRCS := test.cpp
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
LDFLAGS := -lfftw3 -L/opt/homebrew/lib -lomp -larmadillo

# Every folder in ./src will need to be passed to GCC so that it can find header files
INC_DIRS := 
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
INC_FLAGS := $(addprefix -I,$(INC_DIRS)) 

# The -MMD and -MP flags together generate Makefiles for us!
# These files will have .d instead of .o as the output.
CPPFLAGS := $(INC_FLAGS) -MMD -MP -Wall -Werror -Wno-sign-compare -O3 -I/opt/homebrew/include
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
test:
	$(BUILD_DIR)/$(TARGET)

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)
