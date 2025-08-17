 #include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric> // for std::accumulate

#include "block.hpp"
#include "kernels.hpp"

namespace riptide {

struct Candidate {
    size_t x;          // pixel column in full image
    size_t y;          // pixel row in full image
    size_t width_bins; // trial width in bins
    float snr;         // signal-to-noise ratio
    float period;      // the period

    std::vector<float> snr_cutout; // fixed-size cutout
    size_t cut_nx;    // cutout width
    size_t cut_ny;    // cutout height
};

inline std::vector<float> extract_cutout(
    const float* image,
    size_t nx,
    size_t ny,
    size_t cx,
    size_t cy,
    size_t cut_w,
    size_t cut_h)
{
    std::vector<float> cutout(cut_w * cut_h, 0.0f);

    ssize_t half_w = static_cast<ssize_t>(cut_w) / 2;
    ssize_t half_h = static_cast<ssize_t>(cut_h) / 2;

    for (size_t j = 0; j < cut_h; ++j) {
        for (size_t i = 0; i < cut_w; ++i) {
            ssize_t xx = static_cast<ssize_t>(cx) + i - half_w;
            ssize_t yy = static_cast<ssize_t>(cy) + j - half_h;
            if (xx >= 0 && yy >= 0 && xx < static_cast<ssize_t>(nx) && yy < static_cast<ssize_t>(ny)) {
                cutout[j * cut_w + i] = image[yy * nx + xx];
            }
        }
    }
    return cutout;
}

  
/*
 Compute snr for a single 1D phase profile.
 - arr: length nbins (float)
 - widths: array of widths
 - num_widths: number of widths
 - out_snr: length num_widths (written)
*/
inline void snr1(
    const float* __restrict__ arr,
    size_t nbins,
    const size_t* widths,
    size_t num_widths,
    float stdnoise,
    float* __restrict__ out_snr,
    float* __restrict__ cpfsum_buf)
{
    const size_t wmax = *std::max_element(widths, widths + num_widths);

    circular_prefix_sum(arr, nbins, nbins + wmax, cpfsum_buf);
    const float sum = cpfsum_buf[nbins - 1];

    for (size_t iw = 0; iw < num_widths; ++iw) {
        const size_t w = widths[iw];
        const float h = std::sqrt((nbins - w) / float(nbins * w));
        const float b = w / float(nbins - w) * h;

        auto res = diff_max_index(cpfsum_buf + w, cpfsum_buf, nbins);
        const float dmax = res.first;

        const float snr = ((h + b) * dmax - b * sum) / stdnoise;
        out_snr[iw] = snr;
    }
}

/*
Find candidates from a stack of PSF-convolved real-valued images.
Inputs:
 - images: float image stack, layout [phase * (nx*ny) + y*nx + x]
 - nx, ny: image dims
 - nbins: number of phase bins (number of images)
 - widths: array of trial widths (size_t)
 - num_widths: number of widths
 - snr_thresh: threshold (e.g. 8.0f)
 - max_candidates: maximum candidates to return (e.g. 100)
 - mask_radius: radius in pixels to suppress around accepted candidate (e.g. 2)
Returns:
 - std::vector<Candidate> sorted by descending snr (truncated to max_candidates)
*/
inline std::vector<Candidate> find_candidates_from_images_real(
    const float* __restrict__ images,
    size_t nx,
    size_t ny,
    size_t nbins,
    const size_t* widths,
    size_t num_widths,
    float trial_period,
    float snr_thresh = 8.0f,
    size_t max_candidates = 100,
    size_t mask_radius = 2)
{
    const size_t img_size = nx * ny;
    const size_t total_samples = img_size * nbins;
    const size_t CUT_W = 64;
    const size_t CUT_H = 64;    
    float stdnoise = compute_stddev(images, total_samples);
    if (!(stdnoise > 0.f))
        throw std::invalid_argument("Estimated stdnoise <= 0");

    std::vector<float> snr_cube(num_widths * img_size);
    std::vector<float> profile(nbins);
    std::vector<float> tmp_snr(num_widths);
    std::vector<float> cpfsum(nbins + *std::max_element(widths, widths + num_widths));

    // Compute SNR cube
    for (size_t y = 0; y < ny; ++y) {
        for (size_t x = 0; x < nx; ++x) {
            size_t pix_idx = y * nx + x;
            for (size_t p = 0; p < nbins; ++p)
                profile[p] = images[p * img_size + pix_idx];
            snr1(profile.data(), nbins, widths, num_widths, stdnoise, tmp_snr.data(), cpfsum.data());
            for (size_t iw = 0; iw < num_widths; ++iw)
                snr_cube[iw * img_size + pix_idx] = tmp_snr[iw];
        }
    }

    std::vector<Candidate> candidates;
    const float NEG_INF = -std::numeric_limits<float>::max();
    const ssize_t r = static_cast<ssize_t>(mask_radius);

    // Search for candidates
    for (size_t iw = 0; iw < num_widths; ++iw) {
        const size_t width_bins = widths[iw];
        float* plane = snr_cube.data() + iw * img_size;
	
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
		best_val, //snr
		trial_period,
		extract_cutout(plane, nx, ny, bx, by, CUT_W, CUT_H),
		CUT_W,
		CUT_H,
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
