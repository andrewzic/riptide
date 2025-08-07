#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <cstddef>      // size_t
#include <complex>      // std::complex
#include <cstring>      // memcpy

namespace riptide {

// -----------------------------------
// Float Kernels
// -----------------------------------

inline void add(const float* a, const float* b, size_t n, float* out)
{
    for (size_t i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}

inline void fused_rollback_add(const float* a, const float* b, size_t n, size_t shift, float* out)
{
    const size_t p = shift % n;
    const size_t q = n - p;
    add(a, b + p, q, out);
    add(a + q, b, p, out + q);
}

inline void rollback(const float* x, size_t n, size_t shift, float* out)
{
    const size_t p = shift % n;
    const size_t q = n - p;
    std::memcpy(out, x + p, q * sizeof(float));
    std::memcpy(out + q, x, p * sizeof(float));
}

inline void add_scalar(const float* x, size_t n, float a, float* out)
{
    for (size_t i = 0; i < n; ++i)
        out[i] = x[i] + a;
}

inline float diff_max(const float* x, const float* y, size_t n)
{
    float dmax = x[0] - y[0];
    for (size_t i = 1; i < n; ++i)
    {
        float d = x[i] - y[i];
        if (d > dmax)
            dmax = d;
    }
    return dmax;
}

inline void circular_prefix_sum(const float* x, size_t size, size_t nsum, float* out)
{
    double acc = 0.0;
    const size_t jmax = std::min(size, nsum);

    for (size_t j = 0; j < jmax; ++j)
    {
        acc += x[j];
        out[j] = acc;
    }

    if (nsum <= size)
        return;

    const float sumx = acc;
    const size_t q = nsum / size;
    const size_t r = nsum % size;

    for (size_t i = 1; i < q; ++i)
        add_scalar(out, size, i * sumx, out + i * size);

    add_scalar(out, r, q * sumx, out + q * size);
}

// -----------------------------------
// Complex<float> Kernels
// -----------------------------------

inline void add(const std::complex<float>* a, const std::complex<float>* b, size_t n, std::complex<float>* out)
{
    for (size_t i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}

inline void fused_rollback_add(const std::complex<float>* a, const std::complex<float>* b, size_t n, size_t shift, std::complex<float>* out)
{
    const size_t p = shift % n;
    const size_t q = n - p;
    add(a, b + p, q, out);
    add(a + q, b, p, out + q);
}

inline void rollback(const std::complex<float>* x, size_t n, size_t shift, std::complex<float>* out)
{
    const size_t p = shift % n;
    const size_t q = n - p;
    std::memcpy(out, x + p, q * sizeof(std::complex<float>));
    std::memcpy(out + q, x, p * sizeof(std::complex<float>));
}

inline void add_scalar(const std::complex<float>* x, size_t n, std::complex<float> a, std::complex<float>* out)
{
    for (size_t i = 0; i < n; ++i)
        out[i] = x[i] + a;
}

inline float diff_max(const std::complex<float>* x, const std::complex<float>* y, size_t n)
{
    float dmax = std::abs(x[0] - y[0]);
    for (size_t i = 1; i < n; ++i)
    {
        float d = std::abs(x[i] - y[i]);
        if (d > dmax)
            dmax = d;
    }
    return dmax;
}

inline void circular_prefix_sum(const std::complex<float>* x, size_t size, size_t nsum, std::complex<float>* out)
{
    std::complex<double> acc = 0.0;
    const size_t jmax = std::min(size, nsum);

    for (size_t j = 0; j < jmax; ++j)
    {
        acc += x[j];
        out[j] = acc;
    }

    if (nsum <= size)
        return;

    const std::complex<float> sumx = acc;
    const size_t q = nsum / size;
    const size_t r = nsum % size;

    for (size_t i = 1; i < q; ++i)
        add_scalar(out, size, i * sumx, out + i * size);

    add_scalar(out, r, q * sumx, out + q * size);
}

} // namespace riptide

#endif // KERNELS_HPP
