#ifndef TRANSFORMS_HPP
#define TRANSFORMS_HPP

#include <cstddef> // size_t
#include <cstring> // memcpy()

#include "kernels.hpp"
#include "block.hpp"


namespace riptide {
    // Template for merge
    template <typename T>
    void merge(BlockTemplate<const T> thead, BlockTemplate<const T> ttail, BlockTemplate<T> out)
    {
        const size_t m = out.rows;
        const size_t p = out.cols;

        const float kh = (thead.rows - 1.0f) / (m - 1.0f);
        const float kt = (ttail.rows - 1.0f) / (m - 1.0f);

        for (size_t s = 0; s < m; ++s)
        {
            const size_t h = kh * s + 0.5f;
            const size_t t = kt * s + 0.5f;
            const size_t b = s - (h + t);

            fused_rollback_add(thead.rowptr(h), ttail.rowptr(t), p, h + b, out.rowptr(s));
        }
    }

    // Template for transform
    template <typename T>
    void transform(BlockTemplate<const T> input, BlockTemplate<T> temp, BlockTemplate<T> out)
    {
        const size_t m = input.rows;
        const size_t p = input.cols;

        if (m == 2)
        {
            add(input.rowptr(0), input.rowptr(1), p, out.rowptr(0));
            fused_rollback_add(input.rowptr(0), input.rowptr(1), p, 1, out.rowptr(1));
            return;
        }
        else if (m == 1)
        {
            std::memcpy(out.data, input.data, p * sizeof(T));
            return;
        }

        transform(input.head(), out.head(), temp.head());
        transform(input.tail(), out.tail(), temp.tail());

        merge(input.head(), input.tail(), out);
    }

    // Wrapper for raw pointers
    template <typename T>
    BlockTemplate<T> transform(const T* input, size_t rows, size_t cols, T* temp, T* out)
    {
        transform(
            BlockTemplate<const T>(input, rows, cols),
            BlockTemplate<T>(temp, rows, cols),
            BlockTemplate<T>(out, rows, cols)
        );
        return BlockTemplate<T>(out, rows, cols);
    }

    // Float version wrapper
    inline
    Block transform(const float* input, size_t rows, size_t cols, float* temp, float* out)
    {
        return transform<float>(input, rows, cols, temp, out);
    }

    // Complex version wrapper
    inline
    BlockTemplate<std::complex<float>> transform(const std::complex<float>* input, size_t rows, size_t cols, std::complex<float>* temp, std::complex<float>* out)
    {
        return transform<std::complex<float>>(input, rows, cols, temp, out);
    }

} // namespace riptide

#endif // TRANSFORMS_HPP