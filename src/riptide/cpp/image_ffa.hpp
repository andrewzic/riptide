#ifndef IMAGE_FFA_HPP
#define IMAGE_FFA_HPP

#include <vector>
#include <complex>
#include <stdexcept>
#include <string>
#include <fftw3.h>
#include <cstring> // memcpy
#include <algorithm>
#include <cstddef>

#include "image_snr.hpp" // Candidate struct and find_candidates_from_images_real

namespace riptide {

class ImageFFATrial {
public:

    ImageFFATrial(size_t nx_, size_t ny_)
        : nx(nx_), ny(ny_), uvgrid(nx_ * ny_),
          buf(nullptr), buf_fwd(nullptr), plan_ifft(nullptr), plan_fft(nullptr)
    {
        if (nx == 0 || ny == 0)
            throw std::invalid_argument("Image dimensions must be > 0");

        if ((nx % 2 != 0) || (ny % 2 != 0))
            throw std::invalid_argument("Image dimensions must be even for fftshift");

        // Allocate IFFT buffer and create IFFT plan
        buf = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * nx * ny);
        if (!buf) throw std::bad_alloc();

        plan_ifft = fftwf_plan_dft_2d((int)ny, (int)nx, buf, buf, FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!plan_ifft) {
            fftwf_free(buf);
            buf = nullptr;
            throw std::runtime_error("FFTW backward plan creation failed");
        }
    }

    ~ImageFFATrial() {
        if (plan_ifft) fftwf_destroy_plan(plan_ifft);
        if (buf) fftwf_free(buf);

        // destroy forward plan / free forward buffer if they were allocated
        if (plan_fft) fftwf_destroy_plan(plan_fft);
        if (buf_fwd) fftwf_free(buf_fwd);
    }

    // Accept a real-valued PSF image (image-domain), compute and store its FFT
    // Must be called before img_one() or img_all()
    void set_psf(const std::vector<float>& psf_image) {
        if (psf_image.size() != nx * ny)
            throw std::runtime_error("PSF image size does not match nx*ny");

        // allocate forward buffer & plan lazily if not already
        if (!buf_fwd) {
            buf_fwd = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * nx * ny);
            if (!buf_fwd) throw std::bad_alloc();

            plan_fft = fftwf_plan_dft_2d((int)ny, (int)nx, buf_fwd, buf_fwd, FFTW_FORWARD, FFTW_ESTIMATE);
            if (!plan_fft) {
                fftwf_free(buf_fwd);
                buf_fwd = nullptr;
                throw std::runtime_error("FFTW forward plan creation failed");
            }
        }

        // copy PSF into a complex temporary vector (real->complex)
        std::vector<std::complex<float>> tmp(nx * ny);
        for (size_t i = 0; i < nx * ny; ++i) tmp[i] = { psf_image[i], 0.0f };

        // shift DC from centre to (0,0) before forward FFT (NumPy fftshift convention)
        fftshift2D(tmp, nx, ny);

        // copy into forward fftw buffer
        for (size_t i = 0; i < nx * ny; ++i) {
            buf_fwd[i][0] = tmp[i].real();
            buf_fwd[i][1] = tmp[i].imag();
        }

        // execute forward FFT
        fftwf_execute(plan_fft);

        // store psf FFT (complex) for later use
        psf_fft_img.resize(nx * ny);
        for (size_t i = 0; i < nx * ny; ++i) psf_fft_img[i] = { buf_fwd[i][0], buf_fwd[i][1] };

        //fftshift psf_fft_img to match input uv grid (DC at centre)
        fftshift2D(psf_fft_img, nx, ny);

    }

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
        if (psf_fft_img.empty())
            throw std::runtime_error("PSF FFT not set. Call set_psf() before img_one.");

        const size_t M = u_indices.size();
        const size_t img_size = nx * ny;

        // Clear uvgrid
        std::fill(uvgrid.begin(), uvgrid.end(), std::complex<float>(0.0f, 0.0f));

        // Fill uvgrid from blocks, multiply by precomputed PSF FFT at insert time.
        // (u_indices/v_indices are unique per earlier discussion)
        for (size_t i = 0; i < M; ++i) {
            auto block = get_block(i);
            const size_t rows = block.rows;
            const size_t cols = block.cols;

            if (trial_row >= rows || trial_col >= cols) continue;

            std::complex<float> val = block.data[trial_row * cols + trial_col];
            const size_t ix = u_indices[i];
            const size_t iy = v_indices[i];
            if (ix >= nx || iy >= ny) continue;

            // multiply visibility (uv) by psf_fft_img (also in uv domain)
            uvgrid[iy * nx + ix] = val; // * psf_fft_img[iy * nx + ix];
        }

        // Shift DC from centre -> (0,0) so FFTW sees DC at origin
        fftshift2D(uvgrid, nx, ny);

        // copy uvgrid into IFFT buffer
        for (size_t i = 0; i < img_size; ++i) {
            buf[i][0] = uvgrid[i].real();
            buf[i][1] = uvgrid[i].imag();
        }

        // inverse FFT (uv -> image)
        fftwf_execute(plan_ifft);

        // normalize (FFTW leaves transforms unscaled)
        const float scale = 1.0f / static_cast<float>(img_size);

        // copy only the real part into output (we assume image is real-valued)
        // note: buf[i][0] is real part
        for (size_t i = 0; i < img_size; ++i) {
            out_img[i] = buf[i][0] * scale;
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

            // // Find candidates in this trial_row's image stack (real-valued)
            // std::vector<Candidate> candidates = find_candidates_from_images_real(
            //     all_phase_imgs.data(),
            //     nx, ny,
            //     cols, // nbins = cols
            //     widths,
            //     num_widths,
            //     snr_thresh,
            //     max_candidates,
            //     mask_radius
            // );

            // new: handle per-width candidate lists returned by the current image_snr.hpp
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

            // flatten into a single vector
            all_candidates.insert(all_candidates.end(), candidates_per_width.begin(), candidates_per_width.end());
            // for (size_t iw = 0; iw < candidates_per_width.size(); ++iw) {
            //     const auto &vec = candidates_per_width[iw];
            //     all_candidates.insert(all_candidates.end(), vec.begin(), vec.end());
            // }
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
    if (psf_fft_img.empty())
        throw std::runtime_error("PSF FFT not set. Call set_psf() before img_one.");

    const size_t M = u_indices.size();
    const size_t img_size = nx * ny;

    // local uvgrid copy (avoid mutating the member if not desired)
    std::vector<std::complex<float>> local_uvgrid(nx * ny, {0.0f, 0.0f});

    // Fill uvgrid from blocks, multiply by precomputed PSF FFT
    for (size_t i = 0; i < M; ++i) {
        auto block = get_block(i);
        const size_t rows = block.rows;
        const size_t cols = block.cols;

        if (trial_row >= rows || trial_col >= cols) continue;

        std::complex<float> val = block.data[trial_row * cols + trial_col];
        const size_t ix = u_indices[i];
        const size_t iy = v_indices[i];
        if (ix >= nx || iy >= ny) continue;

        local_uvgrid[iy * nx + ix] = val; //* psf_fft_img[iy * nx + ix];
    }

    // Shift DC from centre -> (0,0)
    fftshift2D(local_uvgrid, nx, ny);

    // copy uvgrid into FFTW buffer
    for (size_t i = 0; i < img_size; ++i) {
        buf[i][0] = local_uvgrid[i].real();
        buf[i][1] = local_uvgrid[i].imag();
    }

    // inverse FFT (uv -> image)
    fftwf_execute(plan_ifft);

    // normalize (FFTW leaves transforms unscaled)
    const float scale = 1.0f / static_cast<float>(img_size);

    std::vector<float> out_img(img_size);
    for (size_t i = 0; i < img_size; ++i) {
        out_img[i] = buf[i][0] * scale;
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

    // temp buffers for SNR calc
    std::vector<float> profile(cols);
    std::vector<float> cpfsum(cols + 1);
    float tmp_snr;

    for (size_t trial_row = 0; trial_row < rows; ++trial_row) {
        // Compute images for all trial_cols
        for (size_t trial_col = 0; trial_col < cols; ++trial_col) {
            float* out_img = &all_phase_imgs[trial_col * img_size];
            img_one(get_block, u_indices, v_indices, trial_row, trial_col, out_img);
        }

        // Grab the uvgrid + image for the first phase bin (trial_col=0)
        auto one_result = img_one_test(get_block, u_indices, v_indices,
                                       trial_row, 0);

        // Estimate noise across stack
        float stdnoise = compute_stddev(all_phase_imgs.data(), cols * img_size);

        // Build SNR plane for width=1
        std::vector<float> plane(img_size);
        float best_snr = -1e30f;
        size_t width_one = 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t pix_idx = y * nx + x;
                for (size_t p = 0; p < cols; ++p)
                    profile[p] = all_phase_imgs[p * img_size + pix_idx];
                snr1(profile.data(), cols, &width_one, 1,
                     stdnoise, &tmp_snr, cpfsum.data());
                plane[pix_idx] = tmp_snr;
                if (tmp_snr > best_snr)
                    best_snr = tmp_snr;
            }
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
    // Compute images for all trial_cols in this row
    
    return {std::move(one_result.uvgrid), std::move(one_result.image)};
}

private:
    size_t nx, ny;
    std::vector<std::complex<float>> uvgrid;
    std::vector<std::complex<float>> psf_fft_img;

    fftwf_complex* buf;        // IFFT buffer
    fftwf_complex* buf_fwd;    // forward FFT buffer (created lazily)
    fftwf_plan plan_ifft;      // plan for inverse transform
    fftwf_plan plan_fft;       // plan for forward transform (psf)

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
