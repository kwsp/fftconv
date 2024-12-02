#pragma once

#include <cstddef>
#include <cstdlib>

class Allocator {
public:
  virtual ~Allocator() = default;
  template <typename T> T *allocate(size_t count) {
    return static_cast<T *>(allocateBytes(count * sizeof(T)));
  };
  virtual void deallocate(void *ptr) = 0;

protected:
  virtual void *allocateBytes(size_t size) = 0;
};

class DefaultAllocator final : public Allocator {
protected:
  void *allocateBytes(size_t size) override { return std::malloc(size); }
  void deallocate(void *ptr) override { std::free(ptr); }
};

#ifdef HAS_FFTW
#include <fftw3.h>

class F

#endif
