#include <fftconv/fftconv.hpp>
#include <fftconv/hilbert.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <span>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using fftconv::ConvMode;

template <typename T> std::span<const T> as_span(py::array_t<T> array) {
  return {array.data(), (size_t)array.size()};
}

template <typename T> std::span<T> as_mutable_span(py::array_t<T> array) {
  return {array.mutable_data(), (size_t)array.size()};
}

static ConvMode parseConvMode(const std::string &mode) {
  if (mode == "same")
    return ConvMode::Same;
  if (mode == "full")
    return ConvMode::Full;
  throw std::runtime_error("Unsupported convolution mode: " + mode);
}

template <typename T>
static void convolve_(py::array_t<T> a, py::array_t<T> k, py::array_t<T> out,
                      const std::string &modeStr) {

  py::buffer_info bufIn = a.request();
  py::buffer_info bufK = k.request();
  py::buffer_info bufOut = out.request();

  // error handling
  if (bufIn.ndim != 1 || bufK.ndim != 1 || bufOut.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be one");
  }

  if (bufIn.size < bufK.size) {
    throw std::runtime_error("Kernel size must be smaller than input size");
  }

  // Execute conv
  const auto mode = parseConvMode(modeStr);
  if (mode == fftconv::ConvMode::Same) {
    fftconv::convolve_fftw<T, fftconv::ConvMode::Same>(as_span(a), as_span(k),
                                                       as_mutable_span(out));
  } else {
    fftconv::convolve_fftw<T, fftconv::ConvMode::Full>(as_span(a), as_span(k),
                                                       as_mutable_span(out));
  }
}

// Same API as np.convolve
template <typename T>
static py::array_t<T> convolve(py::array_t<T> a, py::array_t<T> k,
                               const std::string &modeStr) {
  // Same
  py::ssize_t outSize{};

  const auto mode = parseConvMode(modeStr);
  if (mode == fftconv::ConvMode::Same) {
    outSize = a.size();
  } else { // Full
    outSize = a.size() + k.size() - 1;
  }

  py::array_t<T> out(outSize);
  convolve_(a, k, out, modeStr);
  return out;
}

template <typename T>
static void oaconvolve_(py::array_t<T> a, py::array_t<T> k, py::array_t<T> out,
                        const std::string &modeStr) {

  py::buffer_info bufIn = a.request();
  py::buffer_info bufK = k.request();
  py::buffer_info bufOut = out.request();

  // error handling
  if (bufIn.ndim != 1 || bufK.ndim != 1 || bufOut.ndim != 1) {
    throw std::runtime_error("Number of dimensions must be one");
  }

  if (bufIn.size < bufK.size) {
    throw std::runtime_error("Kernel size must be smaller than input size");
  }

  // Execute conv
  const auto mode = parseConvMode(modeStr);
  if (mode == fftconv::ConvMode::Same) {
    fftconv::oaconvolve_fftw<T, fftconv::ConvMode::Same>(as_span(a), as_span(k),
                                                         as_mutable_span(out));
  } else {
    fftconv::oaconvolve_fftw<T, fftconv::ConvMode::Full>(as_span(a), as_span(k),
                                                         as_mutable_span(out));
  }
}

// Same API as np.convolve
template <typename T>
static py::array_t<T> oaconvolve(py::array_t<T> a, py::array_t<T> k,
                                 const std::string &modeStr) {
  // Same
  py::ssize_t outSize{};

  const auto mode = parseConvMode(modeStr);
  if (mode == fftconv::ConvMode::Same) {
    outSize = a.size();
  } else { // Full
    outSize = a.size() + k.size() - 1;
  }

  py::array_t<T> out(outSize);
  oaconvolve_(a, k, out, modeStr);
  return out;
}

template <typename T>
static void hilbert_(py::array_t<T> a, py::array_t<T> out) {
  fftconv::hilbert<T>(as_span(a), as_mutable_span(out));
}

template <typename T> static py::array_t<T> hilbert(py::array_t<T> a) {
  py::array_t<T> out(a.size());
  hilbert_(a, out);
  return out;
}

const char *const convolve_doc = R"delimiter(
Performs convolution using FFTW. API compatible with np.convolve
)delimiter";

const char *const oaconvolve_doc = R"delimiter(
Performs overlap-add convolution using FFTW. API compatible with np.convolve
)delimiter";

const char *const hilbert_doc = R"delimiter(
Performs envelope detection using the Hilbert transform.
Equivalent to `np.abs(signal.hilbert(a))`
)delimiter";

PYBIND11_MODULE(_pyfftconv, m) {
  m.doc() = "Python wrapper for fftconv";
  m.attr("__version__") = FFTCONV_VERSION;

  m.def("convolve", convolve<double>, py::arg("a"), py::arg("k"),
        py::arg("mode") = "full", convolve_doc);
  m.def("convolve", convolve<float>, py::arg("a"), py::arg("k"),
        py::arg("mode") = "full", convolve_doc);
  m.def("convolve_", convolve_<double>, py::arg("a"), py::arg("k"),
        py::arg("out"), py::arg("mode") = "full", convolve_doc);
  m.def("convolve_", convolve_<float>, py::arg("a"), py::arg("k"),
        py::arg("out"), py::arg("mode") = "full", convolve_doc);

  m.def("oaconvolve", oaconvolve<double>, py::arg("a"), py::arg("k"),
        py::arg("mode") = "full", oaconvolve_doc);
  m.def("oaconvolve", oaconvolve<float>, py::arg("a"), py::arg("k"),
        py::arg("mode") = "full", oaconvolve_doc);
  m.def("oaconvolve_", oaconvolve_<double>, py::arg("a"), py::arg("k"),
        py::arg("out"), py::arg("mode") = "full", oaconvolve_doc);
  m.def("oaconvolve_", oaconvolve_<float>, py::arg("a"), py::arg("k"),
        py::arg("out"), py::arg("mode") = "full", oaconvolve_doc);

  m.def("hilbert", hilbert<double>, py::arg("a"), hilbert_doc);
  m.def("hilbert", hilbert<float>, py::arg("a"), hilbert_doc);
  m.def("hilbert_", hilbert_<double>, py::arg("a"), py::arg("out"),
        hilbert_doc);
  m.def("hilbert_", hilbert_<float>, py::arg("a"), py::arg("out"), hilbert_doc);
}