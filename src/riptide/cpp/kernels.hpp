#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <cstddef>
#include <complex>
#include <cstring>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include <utility> // std::pair

namespace riptide {

// ===================================
// Generic Kernels (templated)
// ===================================

template <typename T>
inline void add(const T* a, const T* b, size_t n, T* out)
{
    for (size_t i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}

template <typename T>
inline void fused_rollback_add(const T* a, const T* b, size_t n, size_t shift, T* out)
{
    const size_t p = shift % n;
    const size_t q = n - p;
    add(a, b + p, q, out);
    add(a + q, b, p, out + q);
}

template <typename T>
inline void rollback(const T* x, size_t n, size_t shift, T* out)
{
    const size_t p = shift % n;
    const size_t q = n - p;
    std::memcpy(out, x + p, q * sizeof(T));
    std::memcpy(out + q, x, p * sizeof(T));
}

template <typename T>
inline void add_scalar(const T* x, size_t n, T a, T* out)
{
    for (size_t i = 0; i < n; ++i)
        out[i] = x[i] + a;
}

template <typename T>
inline auto diff_max(const T* x, const T* y, size_t n)
    -> typename std::conditional<std::is_arithmetic<T>::value, T, float>::type
{
    using ReturnT = typename std::conditional<std::is_arithmetic<T>::value, T, float>::type;

    ReturnT dmax = std::is_arithmetic<T>::value ? (x[0] - y[0]) : std::abs(x[0] - y[0]);

    for (size_t i = 1; i < n; ++i)
    {
        ReturnT d = std::is_arithmetic<T>::value ? (x[i] - y[i]) : std::abs(x[i] - y[i]);
        if (d > dmax)
            dmax = d;
    }
    return dmax;
}

template <typename T>
inline void circular_prefix_sum(const T* x, size_t size, size_t nsum, T* out)
{
    using Scalar = typename std::conditional<std::is_arithmetic<T>::value, T, float>::type;

    T acc = T(0);
    const size_t jmax = std::min(size, nsum);

    for (size_t j = 0; j < jmax; ++j)
    {
        acc += x[j];
        out[j] = acc;
    }

    if (nsum <= size)
        return;

    const T sumx = acc;
    const size_t q = nsum / size;
    const size_t r = nsum % size;

    for (size_t i = 1; i < q; ++i)
        add_scalar(out, size, sumx * static_cast<Scalar>(i), out + i * size);

    add_scalar(out, r, sumx * static_cast<Scalar>(q), out + q * size);
}

// ===================================
// diff_max_index helper
// ===================================
template <typename T>
inline std::pair<T, size_t> diff_max_index(const T* x, const T* y, size_t n)
{
    using Scalar = typename std::conditional<std::is_arithmetic<T>::value, T, float>::type;

    Scalar best_val = std::is_arithmetic<T>::value ? (x[0] - y[0]) : std::abs(x[0] - y[0]);
    size_t best_idx = 0;

    for (size_t i = 1; i < n; ++i) {
        Scalar val = std::is_arithmetic<T>::value ? (x[i] - y[i]) : std::abs(x[i] - y[i]);
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }
    return {best_val, best_idx};
}

// Compute stddev of an array
inline float compute_stddev(const float* data, size_t n)
{
    float mean = std::accumulate(data, data + n, 0.f) / n;
    float accum = 0.f;
    for (size_t i = 0; i < n; ++i) {
        float diff = data[i] - mean;
        accum += diff * diff;
    }
    return std::sqrt(accum / n);
}

} // namespace riptide

#endif // KERNELS_HPP
