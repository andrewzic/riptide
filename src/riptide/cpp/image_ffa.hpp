#ifndef IMAGE_FFA_HPP
#define IMAGE_FFA_HPP

#include <vector>
#include <complex>
#include <stdexcept>
#include <string>
#include <cstring> // memcpy
#include <algorithm>
#include <cstddef>
#include <limits>

#include "ducc0/fft/fft.h"
#include "image_snr.hpp"

namespace riptide {

class ImageFFATrial {
public:

    ImageFFATrial(size_t nx_, size_t ny_)
        : nx(nx_), ny(ny_), uvgrid(nx_ * ny_), ifft_out(nx_ * ny_)
    {
        if (nx == 0 || ny == 0)
            throw std::invalid_argument("Image dimensions must be > 0");

        if ((nx % 2 != 0) || (ny % 2 != 0))
            throw std::invalid_argument("Image dimensions must be even for fftshift");
    }

    ~ImageFFATrial() = default;

    // Build one real image for a given (trial_row, trial_col).
    // Writes nx*ny floats to out_img (row-major: y*nx + x).
    template <typename BlockGetter>
    void img_one(const BlockGetter& get_block,
                 const std::vector<size_t>& u_indices,
                 const std::vector<size_t>& v_indices,
                 size_t trial_row,
                 size_t trial_col,
                 float* out_img)
    {
        if (u_indices.size() != v_indices.size())
            throw std::runtime_error("u_indices and v_indices must match in size");

        const size_t M = u_indices.size();
        const size_t img_size = nx * ny;

        // Clear uvgrid
        std::fill(uvgrid.begin(), uvgrid.end(), std::complex<float>(0.0f, 0.0f));

        // Fill uvgrid from blocks.
        // Note: unique_uv from np.nonzero(uv_mask) stores (row_idx, col_idx) = (iy, ix)
        for (size_t i = 0; i < M; ++i) {
            auto block = get_block(i);
            const size_t rows = block.rows;
            const size_t cols = block.cols;

            if (trial_row >= rows || trial_col >= cols) continue;

            std::complex<float> val = block.data[trial_row * cols + trial_col];
            const size_t iy = u_indices[i];
            const size_t ix = v_indices[i];
            if (ix >= nx || iy >= ny) continue;

            uvgrid[iy * nx + ix] = val;
        }

        // Shift DC from centre -> (0,0) so inverse FFT sees DC at origin
        fftshift2D(uvgrid, nx, ny);

        // inverse 2D FFT (uv -> image) using ducc0::c2c
        const float scale = 1.0f / static_cast<float>(img_size);
        ducc0::cfmav<std::complex<float>> in_mav(uvgrid.data(), {ny, nx});
        ducc0::vfmav<std::complex<float>> out_mav(ifft_out.data(), {ny, nx});
        ducc0::c2c(in_mav, out_mav, {0, 1}, false, scale, 1);

        // copy only the real part into output (Hermitian UV grid -> real dirty image)
        for (size_t i = 0; i < img_size; ++i) {
            out_img[i] = ifft_out[i].real();
        }

        // Post-shift: move DC back to centre (NumPy convention)
        fftshift2D(out_img, nx, ny);
    }

    // Process all trial_rows, produce candidate list aggregated over all trial_rows.
    // Returns vector of Candidate (sorted by snr, truncated to max_candidates_all).
    template <typename BlockGetter>
    std::vector<Candidate> img_all(
        const BlockGetter& get_block,
        const std::vector<size_t>& u_indices,
        const std::vector<size_t>& v_indices,
        const size_t* widths,
        size_t num_widths,
        const std::vector<float>& trial_periods,
        float snr_thresh = 8.0f,
        size_t max_candidates = 100,
        size_t max_candidates_all = 1000,
        size_t mask_radius = 2)
    {
        if (u_indices.size() != v_indices.size())
            throw std::runtime_error("u_indices and v_indices must match in size");

        if (u_indices.empty())
            throw std::runtime_error("No blocks provided");

        // check all blocks have same shape
        auto first_block = get_block(0);
        const size_t rows = first_block.rows;
        const size_t cols = first_block.cols;
        for (size_t i = 1; i < u_indices.size(); ++i) {
            auto blk = get_block(i);
            if (blk.rows != rows || blk.cols != cols)
                throw std::runtime_error("All blocks must have the same shape");
        }

        const size_t img_size = nx * ny;

        // Reuse single allocation for phase images for current trial_row:
        // layout: phase-major: [phase=cols][pixel(img_size)]
        std::vector<float> all_phase_imgs(cols * img_size);

        std::vector<Candidate> all_candidates;
        all_candidates.reserve(std::min<size_t>(256, max_candidates_all));

        // Loop over trial_row (periods)
        for (size_t trial_row = 0; trial_row < rows; ++trial_row) {

            // Compute images for all trial_cols (phase bins) for this trial_row
            for (size_t trial_col = 0; trial_col < cols; ++trial_col) {
                float* out_img = &all_phase_imgs[trial_col * img_size];
                img_one(get_block, u_indices, v_indices, trial_row, trial_col, out_img);
            }

            auto candidates_per_width = find_image_candidates(
                all_phase_imgs.data(),
                nx, ny,
                cols, // nbins = cols
                widths,
                num_widths,
                trial_periods[trial_row],
                snr_thresh,
                max_candidates,
                mask_radius
            );

            all_candidates.insert(all_candidates.end(), candidates_per_width.begin(), candidates_per_width.end());
        }
        // global sort/truncate
        std::sort(all_candidates.begin(), all_candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.snr > b.snr; });

        if (all_candidates.size() > max_candidates_all)
            all_candidates.resize(max_candidates_all);

        return all_candidates;
    }

    // Build one real image for a given (trial_row, trial_col).
    // Returns both the UV grid and the real-space image.
    template <typename BlockGetter>
    ImgOneResult img_one_test(const BlockGetter& get_block,
                              const std::vector<size_t>& u_indices,
                              const std::vector<size_t>& v_indices,
                              size_t trial_row,
                              size_t trial_col)
    {
        if (u_indices.size() != v_indices.size())
            throw std::runtime_error("u_indices and v_indices must match in size");

        const size_t M = u_indices.size();
        const size_t img_size = nx * ny;

        // local uvgrid copy
        std::vector<std::complex<float>> local_uvgrid(nx * ny, {0.0f, 0.0f});

        // Fill uvgrid from blocks (unique_uv stores (row_idx, col_idx) = (iy, ix))
        for (size_t i = 0; i < M; ++i) {
            auto block = get_block(i);
            const size_t rows = block.rows;
            const size_t cols = block.cols;

            if (trial_row >= rows || trial_col >= cols) continue;

            std::complex<float> val = block.data[trial_row * cols + trial_col];
            const size_t iy = u_indices[i];
            const size_t ix = v_indices[i];
            if (ix >= nx || iy >= ny) continue;

            local_uvgrid[iy * nx + ix] = val;
        }

        // Shift DC from centre -> (0,0)
        fftshift2D(local_uvgrid, nx, ny);

        // inverse 2D FFT (uv -> image) using ducc0::c2c
        const float scale = 1.0f / static_cast<float>(img_size);
        std::vector<std::complex<float>> local_ifft_out(img_size);
        ducc0::cfmav<std::complex<float>> in_mav(local_uvgrid.data(), {ny, nx});
        ducc0::vfmav<std::complex<float>> out_mav(local_ifft_out.data(), {ny, nx});
        ducc0::c2c(in_mav, out_mav, {0, 1}, false, scale, 1);

        std::vector<float> out_img(img_size);
        for (size_t i = 0; i < img_size; ++i) {
            out_img[i] = local_ifft_out[i].real();
        }

        // Post: move DC back to centre (np convention)
        fftshift2D(out_img, nx, ny);

        return {std::move(local_uvgrid), std::move(out_img)};
    }

    template <typename BlockGetter>
    std::vector<ImgTestResult> img_all_test_img(
        const BlockGetter& get_block,
        const std::vector<size_t>& u_indices,
        const std::vector<size_t>& v_indices,
        const std::vector<float>& trial_periods,
        size_t nx,
        size_t ny)
    {
        if (u_indices.size() != v_indices.size())
            throw std::runtime_error("u_indices and v_indices must match in size");

        if (u_indices.empty())
            throw std::runtime_error("No blocks provided");

        auto first_block = get_block(0);
        const size_t rows = first_block.rows;
        const size_t cols = first_block.cols;
        for (size_t i = 1; i < u_indices.size(); ++i) {
            auto blk = get_block(i);
            if (blk.rows != rows || blk.cols != cols)
                throw std::runtime_error("All blocks must have the same shape");
        }

        const size_t img_size = nx * ny;
        std::vector<float> all_phase_imgs(cols * img_size);

        std::vector<ImgTestResult> results;
        results.reserve(rows);

        // Generate default boxcar width trials for cols phase bins
        std::vector<size_t> widths = generate_default_image_widths(cols, 0.5, 1.5);
        std::vector<float> mean_img(img_size);
        std::vector<float> box_img(img_size);
        std::vector<float> width_plane(img_size);

        for (size_t trial_row = 0; trial_row < rows; ++trial_row) {
            // Compute images for all trial_cols (phase bins)
            for (size_t trial_col = 0; trial_col < cols; ++trial_col) {
                float* out_img = &all_phase_imgs[trial_col * img_size];
                img_one(get_block, u_indices, v_indices, trial_row, trial_col, out_img);
            }

            // Grab the uvgrid + image for the first phase bin (trial_col=0)
            auto one_result = img_one_test(get_block, u_indices, v_indices,
                                           trial_row, 0);

            // Compute hybrid boxcar-matched + spatial RMS SNR plane across width trials
            compute_phase_mean_image(all_phase_imgs.data(), img_size, cols, mean_img);

            const float NEG_INF = -std::numeric_limits<float>::max();
            std::vector<float> plane(img_size, NEG_INF);

            for (size_t iw = 0; iw < widths.size(); ++iw) {
                compute_boxcar_snr_plane(
                    all_phase_imgs.data(),
                    mean_img.data(),
                    img_size,
                    cols,
                    widths[iw],
                    width_plane.data(),
                    box_img
                );
                for (size_t p = 0; p < img_size; ++p) {
                    if (width_plane[p] > plane[p]) {
                        plane[p] = width_plane[p];
                    }
                }
            }

            float best_snr = -1e30f;
            for (size_t p = 0; p < img_size; ++p) {
                if (plane[p] == NEG_INF) plane[p] = 0.0f;
                if (plane[p] > best_snr) best_snr = plane[p];
            }

            results.push_back({
                std::move(plane),
                best_snr,
                trial_periods[trial_row],
                std::move(one_result.uvgrid),
                std::move(one_result.image)
            });
        }

        return results;
    }

    template <typename BlockGetter>
    ImgOneResult img_all_test(
        const BlockGetter& get_block,
        const std::vector<size_t>& u_indices,
        const std::vector<size_t>& v_indices)
    {
        if (u_indices.size() != v_indices.size())
            throw std::runtime_error("u_indices and v_indices must match in size");

        if (u_indices.empty())
            throw std::runtime_error("No blocks provided");

        // check all blocks have same shape
        auto first_block = get_block(0);
        const size_t rows = first_block.rows;
        const size_t cols = first_block.cols;
        for (size_t i = 1; i < u_indices.size(); ++i) {
            auto blk = get_block(i);
            if (blk.rows != rows || blk.cols != cols)
                throw std::runtime_error("All blocks must have the same shape");
        }

        // Only process the first trial_row
        size_t trial_row = 0;
        size_t trial_col = 0;
        auto one_result = img_one_test(get_block, u_indices, v_indices,
                                       trial_row, trial_col);

        return {std::move(one_result.uvgrid), std::move(one_result.image)};
    }

private:
    size_t nx, ny;
    std::vector<std::complex<float>> uvgrid;
    std::vector<std::complex<float>> ifft_out;

    // 2D fftshift for complex vector (row-major)
    static void fftshift2D(std::vector<std::complex<float>>& data, size_t nx, size_t ny) {
        size_t halfx = nx / 2;
        size_t halfy = ny / 2;
        for (size_t y = 0; y < halfy; ++y) {
            for (size_t x = 0; x < halfx; ++x) {
                std::swap(data[y * nx + x], data[(y + halfy) * nx + (x + halfx)]);
                std::swap(data[y * nx + (x + halfx)], data[(y + halfy) * nx + x]);
            }
        }
    }

    // 2D fftshift for real vector (row-major)
    static void fftshift2D(std::vector<float>& data, size_t nx, size_t ny) {
        size_t halfx = nx / 2;
        size_t halfy = ny / 2;
        for (size_t y = 0; y < halfy; ++y) {
            for (size_t x = 0; x < halfx; ++x) {
                const size_t a = y * nx + x;
                const size_t b = (y + halfy) * nx + (x + halfx);
                std::swap(data[a], data[b]);

                const size_t c = y * nx + (x + halfx);
                const size_t d = (y + halfy) * nx + x;
                std::swap(data[c], data[d]);
            }
        }
    }

    // 2D fftshift for raw float pointer (in-place)
    static void fftshift2D(float* data, size_t nx, size_t ny) {
        size_t halfx = nx / 2;
        size_t halfy = ny / 2;
        for (size_t y = 0; y < halfy; ++y) {
            for (size_t x = 0; x < halfx; ++x) {
                const size_t a = y * nx + x;
                const size_t b = (y + halfy) * nx + (x + halfx);
                std::swap(data[a], data[b]);

                const size_t c = y * nx + (x + halfx);
                const size_t d = (y + halfy) * nx + x;
                std::swap(data[c], data[d]);
            }
        }
    }
}; // class ImageFFATrial

} // namespace riptide

#endif // IMAGE_FFA_HPP
