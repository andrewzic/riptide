#ifndef IMAGE_SNR_HPP
#define IMAGE_SNR_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <complex>

#include "block.hpp"
#include "kernels.hpp"

namespace riptide {

struct Candidate {
    size_t x;          // pixel column in full image
    size_t y;          // pixel row in full image
    size_t width_bins; // trial width in bins
    float snr;         // signal-to-noise ratio
    float period;      // the period

    std::vector<float> cutout; // fixed-size cutout
    size_t cut_nx;    // cutout width
    size_t cut_ny;    // cutout height
};

struct ImgOneResult {
    std::vector<std::complex<float>> uvgrid;  // uv-domain grid
    std::vector<float> image;                 // real-space image
};

struct ImgTestResult {
    std::vector<float> snr_plane;      // (ny*nx)
    float best_snr;
    float trial_period;
    std::vector<std::complex<float>> uvgrid; // from img_one_test
    std::vector<float> image;                // from img_one_test
};

inline std::vector<float> extract_cutout(
    const float* image,
    size_t nx,
    size_t ny,
    size_t cx,
    size_t cy,
    size_t cut_nx,
    size_t cut_ny)
{
    std::vector<float> cutout(cut_nx * cut_ny, 0.0f);

    ssize_t half_w = static_cast<ssize_t>(cut_nx) / 2;
    ssize_t half_h = static_cast<ssize_t>(cut_ny) / 2;

    for (size_t j = 0; j < cut_ny; ++j) {
        for (size_t i = 0; i < cut_nx; ++i) {
            ssize_t xx = static_cast<ssize_t>(cx) + i - half_w;
            ssize_t yy = static_cast<ssize_t>(cy) + j - half_h;
            if (xx >= 0 && yy >= 0 && xx < static_cast<ssize_t>(nx) && yy < static_cast<ssize_t>(ny)) {
                cutout[j * cut_nx + i] = image[yy * nx + xx];
            }
        }
    }
    return cutout;
}

/*
 Compute the phase-averaged (continuum) dirty image across all phase bins:
   mean_img[p] = (1 / nbins) * sum_{b=0}^{nbins-1} images[b * img_size + p]
*/
inline void compute_phase_mean_image(
    const float* __restrict__ images,
    size_t img_size,
    size_t nbins,
    std::vector<float>& mean_img)
{
    mean_img.assign(img_size, 0.0f);
    if (nbins == 0 || img_size == 0) return;

    for (size_t b = 0; b < nbins; ++b) {
        const float* __restrict__ img_b = images + b * img_size;
        for (size_t p = 0; p < img_size; ++p) {
            mean_img[p] += img_b[p];
        }
    }

    const float inv_nbins = 1.0f / static_cast<float>(nbins);
    for (size_t p = 0; p < img_size; ++p) {
        mean_img[p] *= inv_nbins;
    }
}

/*
 Hybrid matched-boxcar + spatial RMS S/N plane for a given pulse width (width_bins):
   For each starting phase bin b in [0, nbins):
     boxcar_w(I(b))[p] = sum_{k=0}^{w-1} (I((b + k) % nbins, p) - mean_img[p])
     sigma_xy = stddev_{x,y}(boxcar_w(I(b)))
     SNR_w(b, p) = boxcar_w(I(b))[p] / sigma_xy
   out_plane[p] = max_{b} SNR_w(b, p)

 Note: The zero-mean unit-norm boxcar scale factor (h + b) = sqrt(nbins / (w * (nbins - w)))
 is a positive constant across all spatial pixels (x, y) for fixed (w, b), so it cancels
 identically between the numerator boxcar_w(I(b))[p] and denominator sigma_xy(boxcar_w(I(b))).
*/
inline void compute_boxcar_snr_plane(
    const float* __restrict__ images,
    const float* __restrict__ mean_img,
    size_t img_size,
    size_t nbins,
    size_t width_bins,
    float* __restrict__ out_plane,
    std::vector<float>& box_img)
{
    const float NEG_INF = -std::numeric_limits<float>::max();
    std::fill(out_plane, out_plane + img_size, NEG_INF);

    if (width_bins == 0 || width_bins >= nbins || img_size == 0) {
        std::fill(out_plane, out_plane + img_size, 0.0f);
        return;
    }

    box_img.resize(img_size);
    const float w_f = static_cast<float>(width_bins);
    bool any_valid_phase = false;

    for (size_t b = 0; b < nbins; ++b) {
        // Compute box_img directly for phase bin b to avoid float32 drift when sliding past empty bins:
        for (size_t p = 0; p < img_size; ++p) {
            box_img[p] = -w_f * mean_img[p];
        }
        for (size_t k = 0; k < width_bins; ++k) {
            const float* __restrict__ img_k = images + ((b + k) % nbins) * img_size;
            for (size_t p = 0; p < img_size; ++p) {
                box_img[p] += img_k[p];
            }
        }

        const float sigma_xy = compute_stddev(box_img.data(), img_size);
        if (sigma_xy > 1e-12f) {
            any_valid_phase = true;
            const float inv_sigma = 1.0f / sigma_xy;
            for (size_t p = 0; p < img_size; ++p) {
                const float s = box_img[p] * inv_sigma;
                if (s > out_plane[p]) {
                    out_plane[p] = s;
                }
            }
        }
    }

    if (!any_valid_phase) {
        std::fill(out_plane, out_plane + img_size, 0.0f);
    }
}

/*
 Generate default boxcar width trials in C++ (matching generate_width_trials in Python).
*/
inline std::vector<size_t> generate_default_image_widths(
    size_t nbins,
    double ducy_max = 0.5,
    double wtsp = 1.5)
{
    std::vector<size_t> widths;
    if (nbins <= 1) return widths;

    size_t w = 1;
    size_t wmax = std::max<size_t>(1, static_cast<size_t>(ducy_max * static_cast<double>(nbins)));
    if (wmax >= nbins) {
        wmax = nbins - 1;
    }

    while (w <= wmax && w < nbins) {
        widths.push_back(w);
        size_t next_w = static_cast<size_t>(wtsp * static_cast<double>(w));
        w = std::max(w + 1, next_w);
    }
    return widths;
}

/*
 Find candidates from a stack of real-valued folded phase images using the hybrid
 boxcar-matched + spatial RMS estimator:
   SNR_w(y, x) = max_{b} [ boxcar_w(I(b))(y, x) / sigma_{x,y}(boxcar_w(I(b))) ]
 Inputs:
  - images: float image stack, layout [phase * (nx*ny) + y*nx + x]
  - nx, ny: image dims
  - nbins: number of phase bins (number of images)
  - widths: array of trial widths (size_t)
  - num_widths: number of widths
  - trial_period: trial period (seconds)
  - snr_thresh: threshold (e.g. 8.0f)
  - max_candidates: maximum candidates to return (e.g. 100)
  - mask_radius: radius in pixels to suppress around accepted candidate (e.g. 2)
 Returns:
  - std::vector<Candidate> sorted by descending snr (truncated to max_candidates)
*/
inline std::vector<Candidate> find_image_candidates(
    const float* __restrict__ images,
    size_t nx,
    size_t ny,
    size_t nbins,
    const size_t* widths,
    size_t num_widths,
    float trial_period,
    float snr_thresh = 8.0f,
    size_t max_candidates = 100,
    size_t mask_radius = 2,
    size_t cut_nx = 64,
    size_t cut_ny = 64)
{
    const size_t img_size = nx * ny;
    std::vector<Candidate> candidates;
    if (img_size == 0 || nbins <= 1 || num_widths == 0) {
        return candidates;
    }

    const float NEG_INF = -std::numeric_limits<float>::max();
    const ssize_t r = static_cast<ssize_t>(mask_radius);

    // Precompute phase-mean continuum image once for this trial period
    std::vector<float> mean_img;
    compute_phase_mean_image(images, img_size, nbins, mean_img);

    // Reusable buffers across width trials
    std::vector<float> box_img(img_size);
    std::vector<float> plane(img_size);

    // --- process one width at a time ---
    for (size_t iw = 0; iw < num_widths; ++iw) {
        const size_t width_bins = widths[iw];
        if (width_bins == 0 || width_bins >= nbins) continue;

        compute_boxcar_snr_plane(
            images,
            mean_img.data(),
            img_size,
            nbins,
            width_bins,
            plane.data(),
            box_img
        );

        // --- candidate search on this width plane ---
        while (true) {
            float best_val = snr_thresh;
            size_t best_pix = 0;

            for (size_t p = 0; p < img_size; ++p) {
                if (plane[p] > best_val) {
                    best_val = plane[p];
                    best_pix = p;
                }
            }
            if (best_val <= snr_thresh)
                break;

            const size_t bx = best_pix % nx;
            const size_t by = best_pix / nx;

            candidates.push_back({
                bx,
                by,
                width_bins,
                best_val, // snr
                trial_period,
                extract_cutout(plane.data(), nx, ny, bx, by, cut_nx, cut_ny),
                cut_nx,
                cut_ny,
            });

            // Mask out a small region around the candidate
            for (ssize_t dy = -r; dy <= r; ++dy) {
                ssize_t yy = static_cast<ssize_t>(by) + dy;
                if (yy < 0 || static_cast<size_t>(yy) >= ny) continue;
                for (ssize_t dx = -r; dx <= r; ++dx) {
                    ssize_t xx = static_cast<ssize_t>(bx) + dx;
                    if (xx < 0 || static_cast<size_t>(xx) >= nx) continue;
                    plane[yy * nx + xx] = NEG_INF;
                }
            }
        }
    }

    // Sort by descending SNR
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.snr > b.snr;
              });

    // Truncate if needed
    if (candidates.size() > max_candidates)
        candidates.resize(max_candidates);

    return candidates;
}

} // namespace riptide

#endif // IMAGE_SNR_HPP
