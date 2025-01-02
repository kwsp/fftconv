/**
A C++ FFTW wrapper
 */
#pragma once

#include <cassert>
#include <complex>
#include <cstdlib>
#include <fftw3.h>
#include <span>
#include <type_traits>
#include <unordered_map>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

// NOLINTBEGIN(*-pointer-arithmetic, *-macro-usage, *-const-cast)

namespace fftw {

// const static unsigned int FLAGS = FFTW_ESTIMATE;
const static unsigned int FLAGS = FFTW_EXHAUSTIVE;

// Place this at the beginning of main() and RAII will take care of setting up
// and tearing down FFTW3 (threads and wisdom)
// NOLINTNEXTLINE(*-special-member-functions)
struct WisdomSetup {
  explicit WisdomSetup(bool threadSafe) {
    static bool callSetup = true;
    if (threadSafe && callSetup) {
      fftw_make_planner_thread_safe();
      callSetup = false;
    }
    fftw_import_wisdom_from_filename(".fftw_wisdom");
    fftwf_import_wisdom_from_filename(".fftwf_wisdom");
  }
  ~WisdomSetup() {
    fftw_export_wisdom_to_filename(".fftw_wisdom");
    fftwf_export_wisdom_to_filename(".fftwf_wisdom");
  }
};

template <typename T>
concept Floating = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <Floating T> struct Traits {
  using Real = T;
  using Complex = std::complex<T>;
};
template <> struct Traits<double> {
  using FftwCx = fftw_complex;
  using PlanT = fftw_plan;
  using IODim = fftw_iodim;
  using R2RKind = fftw_r2r_kind;
  using IODim64 = fftw_iodim64;
};
template <> struct Traits<float> {
  using FftwCx = fftwf_complex;
  using PlanT = fftwf_plan;
  using IODim = fftwf_iodim;
  using R2RKind = fftwf_r2r_kind;
  using IODim64 = fftwf_iodim64;
};

template <Floating T> using Complex = Traits<T>::FftwCx;
template <Floating T> using PlanT = Traits<T>::PlanT;
template <Floating T> using IODim = Traits<T>::IODim;
template <Floating T> using R2RKind = Traits<T>::R2RKind;
template <Floating T> using IODim64 = Traits<T>::IODim64;

template <Floating T> struct prefix_;
template <> struct prefix_<double> {
  static constexpr const char *value = "fftw_";
};
template <> struct prefix_<float> {
  static constexpr const char *value = "fftwf_";
};
template <Floating T> inline constexpr const char *prefix = prefix_<T>::value;

// Macros to concatinate prefix to identifier
#define CONCAT(prefix, name) prefix##name
#define COMMA ,
#define TEMPLATIZE(ReturnT, FUNC, PARAMS, PARAMS_CALL)                         \
  template <Floating T> ReturnT FUNC(PARAMS) {                                 \
    if constexpr (std::is_same_v<T, double>) {                                 \
      return CONCAT(fftw_, FUNC)(PARAMS_CALL);                                 \
    } else if constexpr (std::is_same_v<T, float>) {                           \
      return CONCAT(fftwf_, FUNC)(PARAMS_CALL);                                \
    } else {                                                                   \
      static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,     \
                    "Not implemented");                                        \
    }                                                                          \
  }

TEMPLATIZE(void *, malloc, size_t n, n)
TEMPLATIZE(T *, alloc_real, size_t n, n)
TEMPLATIZE(Complex<T> *, alloc_complex, size_t n, n)
TEMPLATIZE(void, free, void *n, n)

TEMPLATIZE(void, destroy_plan, PlanT<T> plan, plan)

#define PLAN_CREATE_METHOD(FUNC, PARAMS, PARAMS_CALL)                          \
  [[nodiscard]] static Plan FUNC(PARAMS) {                                     \
    Plan<T> planner{[&]() {                                                    \
      if constexpr (std::is_same_v<T, double>) {                               \
        return CONCAT(fftw_plan_, FUNC)(PARAMS_CALL);                          \
      } else if constexpr (std::is_same_v<T, float>) {                         \
        return CONCAT(fftwf_plan_, FUNC)(PARAMS_CALL);                         \
      } else {                                                                 \
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,   \
                      "Not supported");                                        \
      }                                                                        \
    }()};                                                                      \
    return planner;                                                            \
  }

#define PLAN_EXECUTE_METHOD(FUNC, PARAMS, PARAMS_CALL)                         \
  void FUNC(PARAMS) const {                                                    \
    assert(plan != nullptr);                                                   \
    if constexpr (std::is_same_v<T, double>) {                                 \
      CONCAT(fftw_, FUNC)(plan, PARAMS_CALL);                                  \
    } else if constexpr (std::is_same_v<T, float>) {                           \
      CONCAT(fftwf_, FUNC)(plan, PARAMS_CALL);                                 \
    } else {                                                                   \
      static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,     \
                    "Not supported");                                          \
    }                                                                          \
  }

template <typename T> struct Plan {
  PlanT<T> plan;

  Plan() = default;
  Plan(const Plan &) = delete;
  Plan(Plan &&) = default;
  Plan &operator=(const Plan &) = delete;
  Plan &operator=(Plan &&) = default;
  explicit Plan(PlanT<T> plan) : plan(std::move(plan)) {}
  ~Plan() {
    if (plan) {
      destroy_plan<T>(plan);
    }
  }

  /**
   * Basic Interface
   */

  /**
   * Complex DFTs
   * https://fftw.org/fftw3_doc/Complex-DFTs.html
   */

  PLAN_CREATE_METHOD(dft_1d,
                     int n COMMA Complex<T> *in COMMA Complex<T> *out
                         COMMA int sign COMMA unsigned int flags,
                     n COMMA in COMMA out COMMA sign COMMA flags)
  PLAN_CREATE_METHOD(dft_2d,
                     int n0 COMMA int n1 COMMA Complex<T> *in COMMA Complex<T>
                         *out COMMA int sign COMMA unsigned int flags,
                     n0 COMMA n1 COMMA in COMMA out COMMA sign COMMA flags)
  PLAN_CREATE_METHOD(
      dft_3d,
      int n0 COMMA int n1 COMMA int n2 COMMA Complex<T> *in COMMA
          Complex<T> *out COMMA int sign COMMA unsigned int flags,
      n0 COMMA n1 COMMA n2 COMMA in COMMA out COMMA sign COMMA flags)
  PLAN_CREATE_METHOD(dft,
                     int rank COMMA int *n COMMA Complex<T> *in COMMA Complex<T>
                         *out COMMA int sign COMMA unsigned int flags,
                     rank COMMA n COMMA in COMMA out COMMA sign COMMA flags)

  /**
   * Real-data DFTs
   * https://fftw.org/fftw3_doc/Real_002ddata-DFTs.html
   */
  PLAN_CREATE_METHOD(dft_r2c_1d,
                     int n COMMA T *in COMMA Complex<T> *out
                         COMMA unsigned int flags,
                     n COMMA in COMMA out COMMA flags)
  PLAN_CREATE_METHOD(dft_r2c_2d,
                     int n0 COMMA int n1 COMMA T *in COMMA Complex<T> *out
                         COMMA unsigned int flags,
                     n0 COMMA n1 COMMA in COMMA out COMMA flags)
  PLAN_CREATE_METHOD(dft_r2c_3d,
                     int n0 COMMA int n1 COMMA int n2 COMMA T *in COMMA
                         Complex<T> *out COMMA unsigned int flags,
                     n0 COMMA n1 COMMA n2 COMMA in COMMA out COMMA flags)
  PLAN_CREATE_METHOD(dft_r2c,
                     int rank COMMA int *n COMMA T *in COMMA Complex<T> *out
                         COMMA unsigned int flags,
                     rank COMMA n COMMA in COMMA out COMMA flags)

  PLAN_CREATE_METHOD(dft_c2r_1d,
                     int n COMMA Complex<T> *in COMMA T *out
                         COMMA unsigned int flags,
                     n COMMA in COMMA out COMMA flags)
  PLAN_CREATE_METHOD(dft_c2r_2d,
                     int n0 COMMA int n1 COMMA Complex<T> *in COMMA T *out
                         COMMA unsigned int flags,
                     n0 COMMA n1 COMMA in COMMA out COMMA flags)
  PLAN_CREATE_METHOD(dft_c2r_3d,
                     int n0 COMMA int n1 COMMA int n2 COMMA Complex<T> *in COMMA
                         T *out COMMA unsigned int flags,
                     n0 COMMA n1 COMMA n2 COMMA in COMMA out COMMA flags)
  PLAN_CREATE_METHOD(dft_c2r,
                     int rank COMMA int *n COMMA Complex<T> *in COMMA T *out
                         COMMA unsigned int flags,
                     rank COMMA n COMMA in COMMA out COMMA flags)
  /**
   * Real-to-Real Transforms
   * https://fftw.org/fftw3_doc/Real_002dto_002dReal-Transforms.html
   */
  PLAN_CREATE_METHOD(r2r_1d,
                     int n COMMA T *in COMMA T *out COMMA R2RKind<T> kind
                         COMMA unsigned int flags,
                     n COMMA in COMMA out COMMA kind COMMA flags)
  PLAN_CREATE_METHOD(
      r2r_2d,
      int n0 COMMA int n1 COMMA T *in COMMA T *out COMMA R2RKind<T> kind0 COMMA
          R2RKind<T>
              kind1 COMMA unsigned int flags,
      n0 COMMA n1 COMMA in COMMA out COMMA kind0 COMMA kind1 COMMA flags)
  PLAN_CREATE_METHOD(r2r_3d,
                     int n0 COMMA int n1 COMMA int n2 COMMA T *in COMMA T *out
                         COMMA R2RKind<T>
                             kind0 COMMA R2RKind<T>
                                 kind1 COMMA R2RKind<T>
                                     kind2 COMMA unsigned int flags,
                     n0 COMMA n1 COMMA n2 COMMA in COMMA out COMMA kind0 COMMA
                         kind1 COMMA kind2 COMMA flags)
  PLAN_CREATE_METHOD(r2r,
                     int rank COMMA const int *n COMMA T *in COMMA T *out
                         COMMA const R2RKind<T> *kind COMMA unsigned int flags,
                     rank COMMA n COMMA in COMMA out COMMA kind COMMA flags)

  /**
   * Advanced Complex DFTs
   * https://fftw.org/fftw3_doc/Advanced-Complex-DFTs.html
   */
  PLAN_CREATE_METHOD(
      many_dft,
      int rank COMMA const int *n COMMA int howmany COMMA Complex<T> *in
          COMMA const int *inembed COMMA int istride COMMA int idist COMMA
              Complex<T> *out COMMA const int *onembed COMMA int ostride
                  COMMA int odist COMMA int sign COMMA unsigned flags,
      rank COMMA n COMMA howmany COMMA in COMMA inembed COMMA istride COMMA
          idist COMMA out COMMA onembed COMMA ostride COMMA odist COMMA sign
              COMMA flags)

  /**
   * Advanced Real-data DFTs
   * https://fftw.org/fftw3_doc/Advanced-Real_002ddata-DFTs.html
   */
  PLAN_CREATE_METHOD(
      many_dft_r2c,
      int rank COMMA const int *n COMMA int howmany COMMA T *in
          COMMA const int *inembed COMMA int istride COMMA int idist COMMA
              Complex<T> *out COMMA const int *onembed COMMA int ostride
                  COMMA int odist COMMA unsigned flags,
      rank COMMA n COMMA howmany COMMA in COMMA inembed COMMA istride COMMA
          idist COMMA out COMMA onembed COMMA ostride COMMA odist COMMA flags)
  PLAN_CREATE_METHOD(
      many_dft_c2r,
      int rank COMMA const int *n COMMA int howmany COMMA Complex<T> *in
          COMMA const int *inembed COMMA int istride COMMA int idist COMMA
              T *out COMMA const int *onembed COMMA int ostride COMMA int odist
                  COMMA unsigned flags,
      rank COMMA n COMMA howmany COMMA in COMMA inembed COMMA istride COMMA
          idist COMMA out COMMA onembed COMMA ostride COMMA odist COMMA flags)

  /**
   * Advanced Real-to-real Transforms
   * https://fftw.org/fftw3_doc/Advanced-Real_002dto_002dreal-Transforms.html
   */
  PLAN_CREATE_METHOD(
      many_r2r,
      int rank COMMA const int *n COMMA int howmany COMMA T *in
          COMMA const int *inembed COMMA int istride COMMA int idist COMMA
              T *out COMMA const int *onembed COMMA int ostride COMMA int odist
                  COMMA R2RKind<T> *kind COMMA unsigned flags,
      rank COMMA n COMMA howmany COMMA in COMMA inembed COMMA istride COMMA
          idist COMMA out COMMA onembed COMMA ostride COMMA odist COMMA kind
              COMMA flags)

  /**
   * Guru Complex DFTs
   * https://fftw.org/fftw3_doc/Guru-Complex-DFTs.html
   */
  PLAN_CREATE_METHOD(guru_dft,
                     int rank COMMA const IODim<T> *dims COMMA int howmany_rank
                         COMMA const IODim<T> *howmany_dims COMMA Complex<T> *in
                             COMMA Complex<T> *out COMMA int sign
                                 COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA out COMMA sign COMMA flags)
  PLAN_CREATE_METHOD(
      guru64_dft,
      int rank COMMA const IODim64<T> *dims COMMA int howmany_rank
          COMMA const IODim64<T> *howmany_dims COMMA Complex<T> *in COMMA
              Complex<T> *out COMMA int sign COMMA unsigned flags,
      rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA in COMMA out
          COMMA sign COMMA flags)
  PLAN_CREATE_METHOD(guru_split_dft,
                     int rank COMMA const IODim<T> *dims COMMA int howmany_rank
                         COMMA const IODim<T> *howmany_dims COMMA T *ri COMMA
                             T *ii COMMA T *ro COMMA T *io COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         ri COMMA ii COMMA ro COMMA io COMMA flags)
  PLAN_CREATE_METHOD(
      guru64_split_dft,
      int rank COMMA const IODim64<T> *dims COMMA int howmany_rank
          COMMA const IODim64<T> *howmany_dims COMMA T *ri COMMA T *ii COMMA
              T *ro COMMA T *io COMMA unsigned flags,
      rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA ri COMMA ii
          COMMA ro COMMA io COMMA flags)

  /**
   * Guru Real-data DFTs
   * https://fftw.org/fftw3_doc/Guru-Real_002ddata-DFTs.html
   */
  PLAN_CREATE_METHOD(guru_dft_r2c,
                     int rank COMMA const IODim<T> *dims COMMA int howmany_rank
                         COMMA const IODim<T> *howmany_dims COMMA T *in COMMA
                             Complex<T> *out COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA out COMMA flags);
  PLAN_CREATE_METHOD(guru64_dft_r2c,
                     int rank COMMA const IODim64<T> *dims COMMA int
                         howmany_rank COMMA const IODim64<T> *howmany_dims COMMA
                             T *in COMMA Complex<T> *out COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA out COMMA flags);
  PLAN_CREATE_METHOD(guru_split_dft_r2c,
                     int rank COMMA const IODim<T> *dims COMMA int howmany_rank
                         COMMA const IODim<T> *howmany_dims COMMA T *in COMMA
                             T *ro COMMA T *io COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA ro COMMA io COMMA flags);
  PLAN_CREATE_METHOD(guru64_split_dft_r2c,
                     int rank COMMA const IODim64<T> *dims COMMA int
                         howmany_rank COMMA const IODim64<T> *howmany_dims COMMA
                             T *in COMMA T *ro COMMA T *io COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA ro COMMA io COMMA flags);
  PLAN_CREATE_METHOD(guru_dft_c2r,
                     int rank COMMA const IODim<T> *dims COMMA int howmany_rank
                         COMMA const IODim<T> *howmany_dims COMMA
                             fftw_complex *in COMMA T *out COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA out COMMA flags);
  PLAN_CREATE_METHOD(guru64_dft_c2r,
                     int rank COMMA const IODim64<T> *dims COMMA int
                         howmany_rank COMMA const IODim64<T> *howmany_dims COMMA
                             fftw_complex *in COMMA T *out COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA out COMMA flags);
  PLAN_CREATE_METHOD(guru_split_dft_c2r,
                     int rank COMMA const IODim<T> *dims COMMA int howmany_rank
                         COMMA const IODim<T> *howmany_dims COMMA T *ri COMMA
                             T *ii COMMA T *out COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         ri COMMA ii COMMA out COMMA flags);
  PLAN_CREATE_METHOD(guru64_split_dft_c2r,
                     int rank COMMA const IODim64<T> *dims
                         COMMA int howmany_rank
                             COMMA const IODim64<T> *howmany_dims COMMA T *ri
                                 COMMA T *ii COMMA T *out COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         ri COMMA ii COMMA out COMMA flags);

  /**
   * Guru Real-to-real Transforms
   */
  PLAN_CREATE_METHOD(guru_r2r,
                     int rank COMMA const IODim<T> *dims COMMA int howmany_rank
                         COMMA const IODim<T> *howmany_dims COMMA T *in COMMA
                             T *out COMMA const R2RKind<T> *kind
                                 COMMA unsigned flags,
                     rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA
                         in COMMA out COMMA kind COMMA flags)
  PLAN_CREATE_METHOD(
      guru64_r2r,
      int rank COMMA const IODim64<T> *dims COMMA int howmany_rank
          COMMA const IODim64<T> *howmany_dims COMMA T *in COMMA T *out
              COMMA const R2RKind<T> *kind COMMA unsigned flags,
      rank COMMA dims COMMA howmany_rank COMMA howmany_dims COMMA in COMMA out
          COMMA kind COMMA flags)

  void execute() {
    if constexpr (std::is_same_v<T, double>) {
      fftw_execute(plan);
    } else if constexpr (std::is_same_v<T, float>) {
      fftwf_execute(plan);
    } else {
      static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                    "Not supported");
    }
  }

  /**
   * Array execute interface
   * https://fftw.org/fftw3_doc/New_002darray-Execute-Functions.html#New_002darray-Execute-Functions
   */
  PLAN_EXECUTE_METHOD(execute_dft, const Complex<T> *in COMMA Complex<T> *out,
                      const_cast<Complex<T> *>(in) COMMA out);
  PLAN_EXECUTE_METHOD(execute_split_dft,
                      const T *ri COMMA const T *ii COMMA T *ro COMMA T *io,
                      const_cast<T *>(ri) COMMA const_cast<T *>(ii)
                          COMMA ro COMMA io)
  PLAN_EXECUTE_METHOD(execute_dft_r2c, const T *in COMMA Complex<T> *out,
                      const_cast<T *>(in) COMMA out)
  PLAN_EXECUTE_METHOD(execute_split_dft_r2c,
                      const T *in COMMA T *ro COMMA T *io,
                      const_cast<T *>(in) COMMA ro COMMA io);
  PLAN_EXECUTE_METHOD(execute_dft_c2r, const Complex<T> *in COMMA T *out,
                      const_cast<Complex<T> *>(in) COMMA out)
  PLAN_EXECUTE_METHOD(execute_split_dft_c2r,
                      const T *ri COMMA const T *ii COMMA T *out,
                      const_cast<T *>(ri) COMMA const_cast<T *>(ii) COMMA out)
  PLAN_EXECUTE_METHOD(execute_r2r, const T *in COMMA T *out,
                      plan COMMA const_cast<T *>(in) COMMA out)
};

// In memory cache with key type `Key` and value type `Val`
template <class Key, class Val> auto get_cached(Key key) -> Val * {
  thread_local std::unordered_map<Key, std::unique_ptr<Val>> cache;

  auto &val = cache[key];
  if (val == nullptr) {
    val = std::make_unique<Val>(key);
  }
  return val.get();
}

// In memory cache with key type `Key` and value type `Val`
template <class Key, class Val> auto get_cached_stack(Key key) -> Val & {
  thread_local std::unordered_map<Key, Val> cache;

  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }

  auto [it, success] =
      cache.emplace(std::piecewise_construct, std::forward_as_tuple(key),
                    std::forward_as_tuple(key));

  return it->second;
}

template <typename Child> struct cache_mixin {
  // static auto get(size_t n) -> Child & { return *get_cached<size_t,
  // Child>(n); }
  static auto get(size_t n) -> Child & {
    return get_cached_stack<size_t, Child>(n);
  }
};

template <typename T, bool InPlace = false> struct C2CBuffer {
  using Cx = fftw::Complex<T>;
  Cx *in, *out;
  explicit C2CBuffer(size_t n) {
    if constexpr (InPlace) {
      in = fftw::alloc_complex<T>(n);
      out = in;
    } else {
      in = fftw::alloc_complex<T>(n);
      out = fftw::alloc_complex<T>(n);
    }
  }
  C2CBuffer(const C2CBuffer &) = delete;
  C2CBuffer(C2CBuffer &&) = delete;
  C2CBuffer &operator=(const C2CBuffer &) = delete;
  C2CBuffer &operator=(C2CBuffer &&) = delete;
  ~C2CBuffer() noexcept {
    if (in)
      fftw::free<T>(in);

    if constexpr (!InPlace) {
      if (out)
        fftw::free<T>(out);
    }
  }
};

template <typename T, bool InPlace = false> struct C2CSplitBuffer {
  using Cx = fftw::Complex<T>;
  T *ri, *ii, *ro, *io;
  explicit C2CSplitBuffer(size_t n) {
    if constexpr (InPlace) {
      ri = fftw::alloc_real<T>(n);
      ii = fftw::alloc_real<T>(n);
      ro = ri;
      io = ii;
    } else {
      ri = fftw::alloc_real<T>(n);
      ii = fftw::alloc_real<T>(n);
      ro = fftw::alloc_real<T>(n);
      io = fftw::alloc_real<T>(n);
    }
  }
  C2CSplitBuffer(const C2CSplitBuffer &) = delete;
  C2CSplitBuffer(C2CSplitBuffer &&) = delete;
  C2CSplitBuffer &operator=(const C2CSplitBuffer &) = delete;
  C2CSplitBuffer &operator=(C2CSplitBuffer &&) = delete;
  ~C2CSplitBuffer() noexcept {
    if (ri)
      fftw::free<T>(ri);
    if (ii)
      fftw::free<T>(ii);

    if constexpr (!InPlace) {
      if (ro)
        fftw::free<T>(ro);
      if (io)
        fftw::free<T>(io);
    }
  }
};

template <typename T> struct R2CBuffer {
  using Cx = fftw::Complex<T>;
  T *in;
  Cx *out;
  explicit R2CBuffer(size_t n)
      : in(fftw::alloc_real<T>(n)), out(fftw::alloc_complex<T>(n / 2 + 1)) {}
  R2CBuffer(const R2CBuffer &) = delete;
  R2CBuffer(R2CBuffer &&) = delete;
  R2CBuffer &operator=(const R2CBuffer &) = delete;
  R2CBuffer &operator=(R2CBuffer &&) = delete;
  ~R2CBuffer() noexcept {
    if (in)
      fftw::free<T>(in);
    if (out)
      fftw::free<T>(out);
  }
};

template <typename T> struct R2CSplitBuffer {
  using Cx = fftw::Complex<T>;
  T *in, *ro, *io;
  explicit R2CSplitBuffer(size_t n)
      : in(fftw::alloc_real<T>(n)), ro(fftw::alloc_real<T>(n / 2 + 1)),
        io(fftw::alloc_real<T>(n / 2 + 1)) {}
  R2CSplitBuffer(const R2CSplitBuffer &) = delete;
  R2CSplitBuffer(R2CSplitBuffer &&) = delete;
  R2CSplitBuffer &operator=(const R2CSplitBuffer &) = delete;
  R2CSplitBuffer &operator=(R2CSplitBuffer &&) = delete;
  ~R2CSplitBuffer() noexcept {
    if (in)
      fftw::free<T>(in);
    if (ro)
      fftw::free<T>(ro);
    if (io)
      fftw::free<T>(io);
  }
};

template <Floating T, bool InPlace = false>
struct EngineDFT1D : public cache_mixin<EngineDFT1D<T>> {
  using Cx = fftw::Complex<T>;
  using Plan = fftw::Plan<T>;

  C2CBuffer<T> buf;
  Plan plan_forward;
  Plan plan_backward;

  explicit EngineDFT1D(size_t n)
      : buf(n),
        plan_forward(Plan::dft_1d(n, buf.in, buf.out, FFTW_FORWARD, FLAGS)),
        plan_backward(Plan::dft_1d(n, buf.out, buf.in, FFTW_BACKWARD, FLAGS)){};

  void forward() { plan_forward.execute(); }
  void forward(const Cx *in, Cx *out) const { plan_forward.execute(in, out); }
  void backward() { plan_backward.execute(); }
  void backward(const Cx *in, Cx *out) const { plan_backward.execute(in, out); }
};

template <Floating T, bool InPlace = false>
struct EngineDFTSplit1D : public cache_mixin<EngineDFTSplit1D<T, InPlace>> {
  using Cx = fftw::Complex<T>;
  using Plan = fftw::Plan<T>;

  C2CSplitBuffer<T> buf;
  IODim<T> dim;
  Plan plan_forward;
  Plan plan_backward;

  explicit EngineDFTSplit1D(size_t n)
      : buf(n), dim(IODim<T>{.n = static_cast<int>(n), .is = 1, .os = 1}),
        plan_forward(Plan::guru_split_dft(1, &dim, 0, nullptr, buf.ri, buf.ii,
                                          buf.ro, buf.io, FLAGS)),
        plan_backward(Plan::guru_split_dft(1, &dim, 0, nullptr, buf.io, buf.ro,
                                           buf.ii, buf.ri, FLAGS)){
            /*
            https://fftw.org/fftw3_doc/Guru-Complex-DFTs.html#Guru-Complex-DFTs
            There is no sign parameter in fftw_plan_guru_split_dft. This
            function always plans for an FFTW_FORWARD transform. To plan for an
            FFTW_BACKWARD transform, you can exploit the identity that the
            backwards DFT is equal to the forwards DFT with the real and
            imaginary parts swapped.
            */
        };

  void forward() { plan_forward.execute(); }
  void forward(const T *ri, const T *ii, T *ro, T *io) const {
    plan_forward.execute_split_dft(ri, ii, ro, io);
  }
  void backward() { plan_backward.execute(); }
  void backward(const T *ro, const T *io, T *ri, T *ii) const {
    plan_backward.execute_split_dft(ro, io, ri, ii);
  }
};

template <Floating T> struct EngineR2C1D : public cache_mixin<EngineR2C1D<T>> {
  using Cx = fftw::Complex<T>;
  using Plan = fftw::Plan<T>;

  R2CBuffer<T> buf;
  Plan plan_forward;
  Plan plan_backward;

  explicit EngineR2C1D(size_t n)
      : buf(n), plan_forward(Plan::dft_r2c_1d(static_cast<int>(n), buf.in,
                                              buf.out, FLAGS)),
        plan_backward(
            Plan::dft_c2r_1d(static_cast<int>(n), buf.out, buf.in, FLAGS)) {}

  void forward() { plan_forward.execute(); }
  void forward(const T *in, Cx *out) const {
    plan_forward.execute_dft_r2c(in, out);
  }
  void backward() { plan_backward.execute(); }
  void backward(const Cx *in, T *out) const {
    plan_backward.execute_dft_c2r(in, out);
  }
};

/**
Helper functions
 */

/**
out[i] += in[i] * fct
 */
template <typename T>
inline void normalize_add(T *out, T *in, size_t len, T fct) {
  for (size_t i = 0; i < len; ++i) {
    out[i] += in[i] * fct;
  }
}

/**
out[i] += in[i] * fct
 */

template <typename T>
void normalize_add(std::span<T> out, std::span<const T> in, T fct) {
  const auto len = std::min(out.size(), in.size());
  for (size_t i = 0; i < len; ++i) {
    out[i] += in[i] * fct;
  }
}

/**
in[i] *= fct
 */
template <typename t> inline void normalize(t *in, size_t len, t fct) {
  for (size_t i = 0; i < len; ++i) {
    in[i] *= fct;
  }
}

// Take the magnitude of a fftw complex array
// out = sqrt( in.re^2 + in.im^2 )
template <Floating T> void magnitude(Complex<T> const *in, T *out, size_t len) {
  size_t i = 0;
  for (; i < len; ++i) {
    const auto re = in[i][0];
    const auto im = in[i][1];
    out[i] = std::sqrt(re * re + im * im);
  }
}

// Take the magnitude of a split complex array
// out = sqrt( in.re^2 + in.im^2 )
template <Floating T>
void magnitude(T const *ri, T const *ii, T *out, size_t len) {
  size_t i = 0;
  for (; i < len; ++i) {
    const auto re = ri[i];
    const auto im = ii[i];
    out[i] = std::sqrt(re * re + im * im);
  }
}

#if defined(__AVX2__)

// Normalize `in` by `fct` and take the magnitude of a fftw complex array and
// scale `in` by `fct out = sqrt( in.re^2 + in.im^2 )
template <Floating T>
void scale_and_magnitude_avx2(Complex<T> const *in, T *out, size_t const n,
                              const T fct) {
  size_t i = 0;
  constexpr size_t simd_width = 256 / (8 * sizeof(T));

  if constexpr (std::is_same_v<T, float>) {
    const auto fct_vec = _mm256_set1_ps(fct);

    for (; i + simd_width <= n; i += simd_width) {
      auto re =
          _mm256_set_ps(in[i + 7][0], in[i + 6][0], in[i + 5][0], in[i + 4][0],
                        in[i + 3][0], in[i + 2][0], in[i + 1][0], in[i][0]);
      auto im =
          _mm256_set_ps(in[i + 7][1], in[i + 6][1], in[i + 5][1], in[i + 4][1],
                        in[i + 3][1], in[i + 2][1], in[i + 1][1], in[i][1]);

      // scale real and imag
      re = _mm256_mul_ps(re, fct_vec);
      im = _mm256_mul_ps(im, fct_vec);

      // Square the im and re
      auto re2 = _mm256_mul_ps(re, re);
      auto sum2 = _mm256_fmadd_ps(im, im, re2);
      // sqrt
      auto mag = _mm256_sqrt_ps(sum2);

      // store
      _mm256_store_ps(&out[i], mag);
    }

  } else if constexpr (std::is_same_v<T, double>) {
    const auto fct_vec = _mm256_set1_pd(fct);

    for (; i + simd_width <= n; i += simd_width) {
      auto re =
          _mm256_set_pd(in[i + 3][0], in[i + 2][0], in[i + 1][0], in[i][0]);
      auto im =
          _mm256_set_pd(in[i + 3][1], in[i + 2][1], in[i + 1][1], in[i][1]);

      // scale real and imag
      re = _mm256_mul_pd(re, fct_vec);
      im = _mm256_mul_pd(im, fct_vec);

      // Square the im and re
      auto re2 = _mm256_mul_pd(re, re);
      auto sum2 = _mm256_fmadd_pd(im, im, re2);
      // sqrt
      auto mag = _mm256_sqrt_pd(sum2);

      // store
      _mm256_store_pd(&out[i], mag);
    }
  }

  for (; i < n; ++i) {
    const auto re = in[i][0] * fct;
    const auto im = in[i][1] * fct;
    out[i] = std::sqrt(re * re + im * im);
  }
}

#endif

template <Floating T>
void scale_and_magnitude_serial(Complex<T> const *in, T *out, size_t const len,
                                const T fct) {
  size_t i = 0;
  for (; i < len; ++i) {
    const auto re = in[i][0] * fct;
    const auto im = in[i][1] * fct;
    out[i] = std::sqrt(re * re + im * im);
  }
}

// Normalize `in` by `fct` and take the magnitude of a fftw complex array and
// scale `in` by `fct out = sqrt( in.re^2 + in.im^2 )
template <Floating T>
void scale_and_magnitude(Complex<T> const *in, T *out, size_t const len,
                         const T fct) {

#if defined(__AVX2__)

  scale_and_magnitude_avx2(in, out, len, fct);

#else

  scale_and_magnitude_serial(in, out, len, fct);

#endif
}

// Take the magnitude of a split complex array
// out = sqrt( in.re^2 + in.im^2 )
template <Floating T>
void scale_and_magnitude(T const *ri, T const *ii, T *out, size_t const len,
                         const T fct) {
  size_t i = 0;
  for (; i < len; ++i) {
    const auto re = ri[i] * fct;
    const auto im = ii[i] * fct;
    out[i] = std::sqrt(re * re + im * im);
  }
}

template <typename T>
void scale_imag_and_magnitude(T const *real, T const *imag, T fct, size_t n,
                              T *out);

#if defined(__ARM_NEON__)

template <typename T>
void scale_imag_and_magnitude_neon(T const *real, T const *imag, T fct,
                                   size_t n, T *out) {

  size_t i = 0;
  if constexpr (std::is_same_v<T, float>) {

    float32x4_t fct_vec = vdupq_n_f32(fct);
    constexpr size_t simd_width = 4;
    for (; i + simd_width <= n; i += simd_width) {
      auto real_vec = vld1q_f32(&real[i]);
      auto imag_vec = vld1q_f32(&imag[i]);

      // Proc first set
      imag_vec = vmulq_f32(imag_vec, fct_vec);
      real_vec = vmulq_f32(real_vec, real_vec);
      real_vec = vfmaq_f32(real_vec, imag_vec, imag_vec);
      auto res = vsqrtq_f32(real_vec);
      vst1q_f32(&out[i], res);
    }

  } else if constexpr (std::is_same_v<T, double>) {

    float64x2_t fct_vec = vdupq_n_f64(fct);
    constexpr size_t simd_width = 2;
    for (; i + simd_width <= n; i += simd_width) {
      auto real_vec = vld1q_f64(&real[i]);
      auto imag_vec = vld1q_f64(&imag[i]);

      imag_vec = vmulq_f64(imag_vec, fct_vec);
      real_vec = vmulq_f64(real_vec, real_vec);
      real_vec = vfmaq_f64(real_vec, imag_vec, imag_vec);
      auto res = vsqrtq_f64(real_vec);
      vst1q_f64(&out[i], res);
    }
  }

  // Remaining
  for (; i < n; ++i) {
    const auto ri = real[i];
    const auto ii = imag[i] * fct;
    const auto res = std::sqrt(ri * ri + ii * ii);
    out[i] = res;
  }
}

#endif

#if defined(__AVX2__)

template <typename T>
void scale_imag_and_magnitude_avx2(T const *real, T const *imag, T fct,
                                   size_t n, T *out) {
  constexpr size_t prefetch_distance = 16;
  size_t i = 0;

  if constexpr (std::is_same_v<T, float>) {

    constexpr size_t simd_width = 256 / (8 * sizeof(float));
    const auto fct_vec = _mm256_set1_ps(fct);

    for (; i + simd_width <= n; i += simd_width) {
      if (i + prefetch_distance < n) {
        _mm_prefetch((const char *)&real[i + prefetch_distance], _MM_HINT_T0);
        _mm_prefetch((const char *)&imag[i + prefetch_distance], _MM_HINT_T0);
        _mm_prefetch((const char *)&out[i + prefetch_distance], _MM_HINT_T0);
      }

      auto r_vec = _mm256_load_ps(&real[i]);
      auto i_vec = _mm256_load_ps(&imag[i]);

      i_vec = _mm256_mul_ps(i_vec, fct_vec);
      r_vec = _mm256_mul_ps(r_vec, r_vec);
      r_vec = _mm256_fmadd_ps(i_vec, i_vec, r_vec);
      auto res = _mm256_sqrt_ps(r_vec);
      _mm256_store_ps(&out[i], res);
    }

  } else if constexpr (std::is_same_v<T, double>) {
    constexpr size_t simd_width = 256 / (8 * sizeof(double));
    const auto fct_vec = _mm256_set1_pd(fct);
    constexpr size_t prefetch_distance = 16;

    for (; i + simd_width <= n; i += simd_width) {
      if (i + prefetch_distance < n) {
        _mm_prefetch((const char *)&real[i + prefetch_distance], _MM_HINT_T0);
        _mm_prefetch((const char *)&imag[i + prefetch_distance], _MM_HINT_T0);
        _mm_prefetch((const char *)&out[i + prefetch_distance], _MM_HINT_T0);
      }

      auto r_vec = _mm256_load_pd(&real[i]);
      auto i_vec = _mm256_load_pd(&imag[i]);
      i_vec = _mm256_mul_pd(i_vec, fct_vec);
      r_vec = _mm256_mul_pd(r_vec, r_vec);
      r_vec = _mm256_fmadd_pd(i_vec, i_vec, r_vec);
      auto res = _mm256_sqrt_pd(r_vec);
      _mm256_store_pd(&out[i], res);
    }
  }

  for (; i < n; ++i) {
    const auto ri = real[i];
    const auto ii = imag[i] * fct;
    const auto res = std::sqrt(ri * ri + ii * ii);
    out[i] = res;
  }
}

#endif

template <typename T>
void scale_imag_and_magnitude(T const *real, T const *imag, T fct, size_t n,
                              T *out) {

#if defined(__ARM_NEON__)

  scale_imag_and_magnitude_neon<T>(real, imag, fct, n, out);

#elif defined(__AVX2__)

  scale_imag_and_magnitude_avx2<T>(real, imag, fct, n, out);

#else

  for (auto i = 0; i < n; ++i) {
    const auto ri = real[i];
    const auto ii = imag[i] * fct;
    const auto res = std::sqrt(ri * ri + ii * ii);
    out[i] = res;
  }

#endif
}

} // namespace fftw

// NOLINTEND(*-pointer-arithmetic, *-macro-usage, *-const-cast)