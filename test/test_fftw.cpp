#include "test_common.hpp"
#include <array>
#include <fftconv/fftw.hpp>
#include <gtest/gtest.h>

// NOLINTBEGIN(*-magic-numbers, *-pointer-arithmetic, *-non-private-member-*,
// *-member-function, *-destructor)

template <typename T>
void ExpectArraysNear(std::span<const T> arr1, std::span<const T> arr2,
                      T tolerance) {
  assert(arr1.size() == arr2.size());
  return ExpectArraysNear<T>(arr1.data(), arr2.data(), arr1.size(), tolerance);
}

class FFTWPlanCreateC2C : public testing::Test {
protected:
  using T = double;
  fftw::Complex<T> *in;
  fftw::Complex<T> *out;
  int n = 20;

  FFTWPlanCreateC2C() {
    in = fftw::alloc_complex<T>(n);
    out = fftw::alloc_complex<T>(n);
  }

  ~FFTWPlanCreateC2C() override {
    fftw::free<T>(in);
    fftw::free<T>(out);
  }
};

class FFTWPlanCreateC2CSplit : public testing::Test {
protected:
  using T = double;
  T *ri;
  T *ii;
  T *ro;
  T *io;
  int n = 20;

  FFTWPlanCreateC2CSplit() {
    ri = fftw::alloc_real<T>(n);
    ii = fftw::alloc_real<T>(n);
    ro = fftw::alloc_real<T>(n);
    io = fftw::alloc_real<T>(n);
  }

  ~FFTWPlanCreateC2CSplit() override {
    fftw::free<T>(ri);
    fftw::free<T>(ii);
    fftw::free<T>(ro);
    fftw::free<T>(io);
  }
};
class FFTWPlanCreateR2C : public testing::Test {
protected:
  using T = double;
  T *in;
  fftw::Complex<T> *out;
  int n = 20;

  FFTWPlanCreateR2C() {
    in = fftw::alloc_real<T>(n);
    out = fftw::alloc_complex<T>(n / 2 + 1);
  }

  ~FFTWPlanCreateR2C() override {
    fftw::free<T>(in);
    fftw::free<T>(out);
  }
};

class FFTWPlanCreateR2CSplit : public testing::Test {
protected:
  using T = double;
  T *in;
  T *ro;
  T *io;
  int n = 20;

  FFTWPlanCreateR2CSplit() {
    in = fftw::alloc_real<T>(n);
    ro = fftw::alloc_real<T>(n / 2 + 1);
    io = fftw::alloc_real<T>(n / 2 + 1);
  }

  ~FFTWPlanCreateR2CSplit() override {
    fftw::free<T>(in);
    fftw::free<T>(ro);
    fftw::free<T>(io);
  }
};

class FFTWPlanCreateR2R : public testing::Test {
protected:
  using T = double;
  T *in;
  T *out;
  int n = 20;

  FFTWPlanCreateR2R() {
    in = fftw::alloc_real<T>(n);
    out = fftw::alloc_real<T>(n);
  }

  ~FFTWPlanCreateR2R() override {
    fftw::free<T>(in);
    fftw::free<T>(out);
  }
};

TEST_F(FFTWPlanCreateC2C, BasicPlan1d) {
  auto plan = fftw::Plan<T>::dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};
TEST_F(FFTWPlanCreateC2C, BasicPlan2d) {
  auto plan = fftw::Plan<T>::dft_2d(1, n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};
TEST_F(FFTWPlanCreateC2C, BasicPlan3d) {
  auto plan =
      fftw::Plan<T>::dft_3d(1, 1, n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};
TEST_F(FFTWPlanCreateC2C, BasicPlan) {
  int rank = 1;
  int dim = n;
  auto plan =
      fftw::Plan<T>::dft(rank, &dim, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};

TEST_F(FFTWPlanCreateR2C, BasicPlan1d) {
  auto plan_forward = fftw::Plan<T>::dft_r2c_1d(n, in, out, FFTW_ESTIMATE);
  ASSERT_NE(plan_forward.plan, nullptr);
  auto plan_backward = fftw::Plan<T>::dft_c2r_1d(n, out, in, FFTW_ESTIMATE);
  ASSERT_NE(plan_backward.plan, nullptr);
};
TEST_F(FFTWPlanCreateR2C, BasicPlan2d) {
  auto plan_forward = fftw::Plan<T>::dft_r2c_2d(1, n, in, out, FFTW_ESTIMATE);
  ASSERT_NE(plan_forward.plan, nullptr);
  auto plan_backward = fftw::Plan<T>::dft_c2r_2d(1, n, out, in, FFTW_ESTIMATE);
  ASSERT_NE(plan_backward.plan, nullptr);
};
TEST_F(FFTWPlanCreateR2C, BasicPlan3d) {
  auto plan_forward =
      fftw::Plan<T>::dft_r2c_3d(1, 1, n, in, out, FFTW_ESTIMATE);
  ASSERT_NE(plan_forward.plan, nullptr);
  auto plan_backward =
      fftw::Plan<T>::dft_c2r_3d(1, 1, n, out, in, FFTW_ESTIMATE);
  ASSERT_NE(plan_backward.plan, nullptr);
};
TEST_F(FFTWPlanCreateR2C, BasicPlan) {
  int rank = 1;
  int dim = n;
  auto planf = fftw::Plan<T>::dft_r2c(rank, &dim, in, out, FFTW_ESTIMATE);
  ASSERT_NE(planf.plan, nullptr);
  auto planb = fftw::Plan<T>::dft_c2r(rank, &dim, out, in, FFTW_ESTIMATE);
  ASSERT_NE(planb.plan, nullptr);
};

TEST_F(FFTWPlanCreateR2R, BasicPlan1d) {
  auto plan = fftw::Plan<T>::r2r_1d(n, in, out, FFTW_DHT, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};
TEST_F(FFTWPlanCreateR2R, BasicPlan2d) {
  auto plan =
      fftw::Plan<T>::r2r_2d(1, n, in, out, FFTW_DHT, FFTW_DHT, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};
TEST_F(FFTWPlanCreateR2R, BasicPlan3d) {
  auto plan = fftw::Plan<T>::r2r_3d(1, 1, n, in, out, FFTW_DHT, FFTW_DHT,
                                    FFTW_DHT, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};
TEST_F(FFTWPlanCreateR2R, BasicPlan) {
  int rank = 1;
  int dim = n;
  auto kind = FFTW_DHT;
  auto plan = fftw::Plan<T>::r2r(rank, &dim, in, out, &kind, FFTW_ESTIMATE);
  ASSERT_NE(plan.plan, nullptr);
};

/*
Test advanced interface
*/
TEST_F(FFTWPlanCreateC2C, AdvancedPlan) {
  int rank = 1;
  int dim = n;
  int howmany = 1;
  int *inembed = &dim;
  int *onembed = &dim;
  auto pf = fftw::Plan<T>::many_dft(rank, &dim, howmany, in, inembed, 1, 0, out,
                                    onembed, 1, 0, FFTW_FORWARD, FFTW_ESTIMATE);
  ASSERT_NE(pf.plan, nullptr);
}

/*
Test advanced interface
*/
TEST_F(FFTWPlanCreateR2C, AdvancedPlan) {
  int rank = 1;
  int dim = n;
  int howmany = 1;
  int *inembed = &dim;
  int *onembed = &dim;
  auto pf = fftw::Plan<T>::many_dft_r2c(rank, &dim, howmany, in, inembed, 1, 0,
                                        out, onembed, 1, 0, FFTW_ESTIMATE);
  ASSERT_NE(pf.plan, nullptr);

  auto pb = fftw::Plan<T>::many_dft_c2r(rank, &dim, howmany, out, onembed, 1, 0,
                                        in, inembed, 1, 0, FFTW_ESTIMATE);
  ASSERT_NE(pb.plan, nullptr);
}
/*
Test advanced interface
*/
TEST_F(FFTWPlanCreateR2R, AdvancedPlan) {
  int rank = 1;
  int dim = n;
  int howmany = 1;
  int *inembed = &dim;
  int *onembed = &dim;
  auto kind = FFTW_DHT;
  auto pf = fftw::Plan<T>::many_r2r(rank, &dim, howmany, in, inembed, 1, 0, out,
                                    onembed, 1, 0, &kind, FFTW_ESTIMATE);
  ASSERT_NE(pf.plan, nullptr);
}

/*
Test guru interface
*/
TEST_F(FFTWPlanCreateC2C, GuruPlan) {
  {
    int rank = 1;
    int howmany = 1;
    fftw::IODim<T> dim{.n = n, .is = n, .os = n};
    fftw::IODim<T> howmany_dim{.n = n, .is = n, .os = n};

    auto pf = fftw::Plan<T>::guru_dft(rank, &dim, howmany, &howmany_dim, in,
                                      out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(pf.plan, nullptr);
  }
  {
    int rank = 1;
    int howmany = 1;
    fftw::IODim64<T> dim{.n = n, .is = n, .os = n};
    fftw::IODim64<T> howmany_dim{.n = n, .is = n, .os = n};

    auto pf = fftw::Plan<T>::guru64_dft(rank, &dim, howmany, &howmany_dim, in,
                                        out, FFTW_FORWARD, FFTW_ESTIMATE);
    ASSERT_NE(pf.plan, nullptr);
  }
}
TEST_F(FFTWPlanCreateC2CSplit, GuruPlanSplit) {
  int rank = 1;
  fftw::IODim<T> dim{.n = n, .is = 1, .os = 1};
  int howmany = 1;
  fftw::IODim<T> howmany_dim{.n = howmany, .is = n, .os = n};

  auto pf = fftw::Plan<T>::guru_split_dft(rank, &dim, howmany, &howmany_dim, ri,
                                          ii, ro, io, FFTW_ESTIMATE);
  ASSERT_NE(pf.plan, nullptr);
}

TEST_F(FFTWPlanCreateC2CSplit, GuruPlanSplitCorrect) {
  int rank = 1;
  fftw::IODim<T> dim{.n = n, .is = 1, .os = 1};
  int howmany = 1;
  fftw::IODim<T> howmany_dim{.n = howmany, .is = n, .os = n};

  alignas(32) std::array<T, 20> ri = {
      0.24800501, 0.19901191, 0.22109176, 0.65214355, 0.13190115,
      0.83696173, 0.36607799, 0.59169134, 0.89522796, 0.03245338,
      0.39925889, 0.93391339, 0.50966463, 0.26965452, 0.61894752,
      0.29961656, 0.4266788,  0.03522679, 0.0617605,  0.57214474};
  alignas(32) std::array<T, 20> ii{};
  alignas(32) std::array<T, 20> ro{};
  alignas(32) std::array<T, 20> io{};

  std::array<T, 20> expect_ro = {
      8.30143212, -1.4786986,  -0.28946925, 0.59119628,  0.17941478,
      0.54434089, -0.66171488, 0.48975843,  0.12947463,  -0.9028664,
      -0.5442037, -0.9028664,  0.12947463,  0.48975843,  -0.66171488,
      0.54434089, 0.17941478,  0.59119628,  -0.28946925, -1.4786986};

  std::array<T, 20> expect_io = {
      0.,          -0.70254131, -0.35119795, 0.43823534,  -0.23602404,
      1.67620125,  -0.42226181, 2.11848431,  -0.83078287, -1.10366614,
      0.,          1.10366614,  0.83078287,  -2.11848431, 0.42226181,
      -1.67620125, 0.23602404,  -0.43823534, 0.35119795,  0.70254131};

  alignas(32) std::array<T, 20> ri_{};
  alignas(32) std::array<T, 20> ro_{};

  auto pf = fftw::Plan<T>::guru_split_dft(rank, &dim, howmany, &howmany_dim,
                                          ri.data(), ii.data(), ro.data(),
                                          io.data(), FFTW_ESTIMATE);
  ASSERT_NE(pf.plan, nullptr);
  pf.execute();

  ExpectArraysNear<T>(ro, expect_ro, 1e-7);
  ExpectArraysNear<T>(io, expect_io, 1e-7);
}

TEST_F(FFTWPlanCreateR2C, GuruPlan) {
  {
    int rank = 1;
    fftw::IODim<T> dim{.n = n, .is = 1, .os = 1};
    int howmany = 1;
    fftw::IODim<T> howmany_dim{.n = n, .is = n, .os = n};

    auto pf = fftw::Plan<T>::guru_dft_r2c(rank, &dim, howmany, &howmany_dim, in,
                                          out, FFTW_ESTIMATE);
    ASSERT_NE(pf.plan, nullptr);

    auto pb = fftw::Plan<T>::guru_dft_c2r(rank, &dim, howmany, &howmany_dim,
                                          out, in, FFTW_ESTIMATE);
    ASSERT_NE(pb.plan, nullptr);
  }
  {
    int rank = 1;
    int howmany = 1;
    fftw::IODim64<T> dim{.n = n, .is = n, .os = n};
    fftw::IODim64<T> howmany_dim{.n = n, .is = n, .os = n};

    auto pf = fftw::Plan<T>::guru64_dft_r2c(rank, &dim, howmany, &howmany_dim,
                                            in, out, FFTW_ESTIMATE);
    ASSERT_NE(pf.plan, nullptr);

    auto pb = fftw::Plan<T>::guru64_dft_c2r(rank, &dim, howmany, &howmany_dim,
                                            out, in, FFTW_ESTIMATE);
    ASSERT_NE(pb.plan, nullptr);
  }
}

TEST_F(FFTWPlanCreateR2CSplit, GuruPlanSplit) {
  int rank = 1;
  fftw::IODim<T> dim{.n = n, .is = 1, .os = 1};
  int howmany = 1;
  fftw::IODim<T> howmany_dim{.n = howmany, .is = n, .os = n};

  auto pf = fftw::Plan<T>::guru_split_dft_r2c(rank, &dim, howmany, &howmany_dim,
                                              in, ro, io, FFTW_ESTIMATE);
  ASSERT_NE(pf.plan, nullptr);

  auto pb = fftw::Plan<T>::guru_split_dft_c2r(rank, &dim, howmany, &howmany_dim,
                                              ro, io, in, FFTW_ESTIMATE);
  ASSERT_NE(pb.plan, nullptr);
}

TEST_F(FFTWPlanCreateR2CSplit, GuruPlanSplitCorrect) {
  int rank = 1;
  fftw::IODim<T> dim{.n = n, .is = 1, .os = 1};
  int howmany = 0; // For a single transform
  fftw::IODim<T> howmany_dim{.n = howmany, .is = 1, .os = 1};

  alignas(32) std::array<T, 20> in = {
      0.24800501, 0.19901191, 0.22109176, 0.65214355, 0.13190115,
      0.83696173, 0.36607799, 0.59169134, 0.89522796, 0.03245338,
      0.39925889, 0.93391339, 0.50966463, 0.26965452, 0.61894752,
      0.29961656, 0.4266788,  0.03522679, 0.0617605,  0.57214474};

  std::array<T, 11> ro{};
  std::array<T, 11> io{};

  std::array<T, 11> expect_ro = {
      8.30143212,  -1.4786986, -0.28946925, 0.59119628, 0.17941478, 0.54434089,
      -0.66171488, 0.48975843, 0.12947463,  -0.9028664, -0.5442037};

  std::array<T, 11> expect_io = {0.,          -0.70254131, -0.35119795,
                                 0.43823534,  -0.23602404, 1.67620125,
                                 -0.42226181, 2.11848431,  -0.83078287,
                                 -1.10366614, 0.};

  std::array<T, 20> in_{};

  auto pf = fftw::Plan<T>::guru_split_dft_r2c(rank, &dim, howmany, &howmany_dim,
                                              in.data(), ro.data(), io.data(),
                                              FFTW_ESTIMATE);
  ASSERT_NE(pf.plan, nullptr);
  pf.execute();

  ExpectArraysNear<T>(expect_ro, ro, 1e-7);
  ExpectArraysNear<T>(expect_io, io, 1e-7);

  auto pb = fftw::Plan<T>::guru_split_dft_c2r(rank, &dim, howmany, &howmany_dim,
                                              ro.data(), io.data(), in_.data(),
                                              FFTW_ESTIMATE);
  ASSERT_NE(pb.plan, nullptr);
  pb.execute();

  T fct = 1.0 / in_.size();
  for (double &v : in_) {
    v *= fct;
  }
  ExpectArraysNear<T>(in, in_, 1e-7);
}

// NOLINTEND(*-magic-numbers, *-pointer-arithmetic, *-non-private-member-*,
// *-member-function, *-destructor)

int main(int argc, char **argv) {
  fftw::WisdomSetup wisdom(false);
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
