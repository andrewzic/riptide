#ifndef PERIODOGRAM_HPP
#define PERIODOGRAM_HPP

#include <cstddef> // size_t
#include <cstdint>
#include <cmath>
#include <memory>
#include <complex>

#include "downsample.hpp"
#include "transforms.hpp"
#include "block.hpp"
#include "snr.hpp"


namespace riptide {


void periodogram_check_arg(bool condition, std::string const& errmsg)
    {
    if (!(condition))
        throw std::invalid_argument(errmsg);
    }


void periodogram_check_arguments(
    size_t size, 
    double tsamp,
    double period_min, 
    double period_max, 
    size_t bins_min, 
    size_t bins_max)
    {    
    periodogram_check_arg(tsamp > 0, "tsamp must be > 0");
    periodogram_check_arg(period_min > 0, "period_min must be > 0");
    periodogram_check_arg(period_max > period_min, "period_max must be > period_min");
    periodogram_check_arg(bins_min > 1, "bins_min must be > 1");
    periodogram_check_arg(bins_max >= bins_min, "bins_max must be >= bins_min");
    periodogram_check_arg(period_min >= tsamp * bins_min, "Must have: period_min >= tsamp * bins_min ");
    // NOTE: we don't check period_max, the search will automatically stop when the maximum allowable trial period is reached
    }


/*
Returns the first shift in an FFA transform that corresponds to a trial period
equal to, or greater than pmax. pmax must be expressed in units of the 
sampling interval.

This function is useful to calculate how many rows of an FFA transform should
be evaluated for S/N, since often we wish to only consider the rows that
correspond to period trials smaller than p + 1. The index of the last row
to evaluate is equal to ceilshift - 1, or equivalently, the total number
of rows to evaluate is equal to ceilshift.
*/
size_t ceilshift(size_t rows, size_t cols, double pmax)
    {
    return ceil(cols * (rows - 1.0) * (1.0 - cols / pmax));
    }


/*
Returns the total number of trial periods in a periodogram
*/
size_t periodogram_length(
    size_t size,
    double tsamp,
    double period_min,
    double period_max,
    size_t bins_min,
    size_t bins_max)
    {
    periodogram_check_arguments(size, tsamp, period_min, period_max, bins_min, bins_max);

    // Initial downsampling factor
    // We want: ds_ini * tsamp * bmin = period_min
    double ds_ini = period_min / (tsamp * bins_min);

    // Geometric growth factor for the downsampling factor
    double ds_geo = (bins_max + 1.0) / bins_min;

    // Number of required downsampling cycles
    size_t num_downsamplings = ceil(log(period_max / period_min) / log(ds_geo));
    size_t length = 0; // total number of period trials, to be calculated

    /* Downsampling loop */
    for (size_t ids = 0; ids < num_downsamplings; ++ids)
        {
        const double f = ds_ini * pow(ds_geo, ids); // current downsampling factor
        const double tau = f * tsamp; // current sampling time
        const double period_max_samples = period_max / tau;
        const size_t n = downsampled_size(size, f); // current number of input samples

        // Min and max number of bins with which to FFA transform in order to
        // cover all trial periods between period_min and period_max.
        // NOTE: bstop is INclusive
        // Also, we MUST enforce bstop <= n, to avoid doing an FFA transform with 0 rows
        const size_t bstart = bins_min;
        const size_t bstop = std::min({ bins_max, n, size_t(period_max_samples) });

        /* FFA transform loop */
        for (size_t bins = bstart; bins <= bstop; ++bins)
            {
            const size_t rows = n / bins;
            const double period_ceil = std::min(period_max_samples, bins + 1.0);
            const size_t rows_eval = std::min(rows, ceilshift(rows, bins, period_ceil));
            length += rows_eval;
            }
        }
    return length;
    }

/*
Returns the total number of trial periods in a periodogram
*/
size_t periodogram_base_period_length(
    size_t size,
    double tsamp,
    double period_min,
    double period_max,
    size_t bins_min,
    size_t bins_max)
    {
    periodogram_check_arguments(size, tsamp, period_min, period_max, bins_min, bins_max);

    // Initial downsampling factor
    // We want: ds_ini * tsamp * bmin = period_min
    double ds_ini = period_min / (tsamp * bins_min);

    // Geometric growth factor for the downsampling factor
    double ds_geo = (bins_max + 1.0) / bins_min;

    // Number of required downsampling cycles
    size_t num_downsamplings = ceil(log(period_max / period_min) / log(ds_geo));
    size_t length = 0; // total number of period trials, to be calculated

    /* Downsampling loop */
    for (size_t ids = 0; ids < num_downsamplings; ++ids)
        {
        const double f = ds_ini * pow(ds_geo, ids); // current downsampling factor
        const double tau = f * tsamp; // current sampling time
        const double period_max_samples = period_max / tau;
        const size_t n = downsampled_size(size, f); // current number of input samples

        // Min and max number of bins with which to FFA transform in order to
        // cover all trial periods between period_min and period_max.
        // NOTE: bstop is INclusive
        // Also, we MUST enforce bstop <= n, to avoid doing an FFA transform with 0 rows
        const size_t bstart = bins_min;
        const size_t bstop = std::min({ bins_max, n, size_t(period_max_samples) });

        /* FFA transform loop */
        for (size_t bins = bstart; bins <= bstop; ++bins)
            {
            length += 1;
            // #1 extra base period FFA transform per base_period=bins
            }
        }
    return length;
    }


/*
Compute the periodogram of a time series that has been normalised to zero mean and unit variance.
Outputs are: trial periods (num_periods elements), number of phase bins used in the fold (num_periods elements), 
and signal to noise ratio (num_periods * num_widths elements)
*/
void periodogram(
    const float* __restrict__ data,
    size_t size,
    double tsamp,
    const size_t* __restrict__ widths,
    size_t num_widths,
    double period_min,
    double period_max,
    size_t bins_min,
    size_t bins_max,
    double* __restrict__ periods,
    uint32_t* __restrict__ foldbins,
    float* __restrict__ snr)
    {
//     periodogram_check_arguments(size, tsamp, period_min, period_max, bins_min, bins_max);

//     // Initial downsampling factor
//     // We want: ds_ini * tsamp * bmin = period_min
//     double ds_ini = period_min / (tsamp * bins_min);

//     // Geometric growth factor for the downsampling factor
//     double ds_geo = (bins_max + 1.0) / bins_min;

//     // Number of required downsampling cycles
//     size_t num_downsamplings = ceil(log(period_max / period_min) / log(ds_geo));

//     // Allocate buffers
//     const size_t bufsize = downsampled_size(size, ds_ini);
//     std::unique_ptr<float[]> input_mem(new float[bufsize]);
//     std::unique_ptr<float[]> ffabuf_mem(new float[bufsize]);
//     std::unique_ptr<float[]> ffaout_mem(new float[bufsize]);
//     const float* input = input_mem.get();
//     float* ffabuf = ffabuf_mem.get();
//     float* ffaout = ffaout_mem.get();
        
//     /* Downsampling loop */
//     for (size_t ids = 0; ids < num_downsamplings; ++ids)
//         {
//         const double f = ds_ini * pow(ds_geo, ids); // current downsampling factor
//         const double tau = f * tsamp; // current sampling time
//         const double period_max_samples = period_max / tau;
//         const size_t n = downsampled_size(size, f); // current number of input samples

//         // downsample() requires f > 1, but we still allow searching the data at their
//         // original resolution.
//         if (f == 1) {
//             input = data;
//         }            
//         else {
//             downsample(data, size, f, input_mem.get());
//             input = input_mem.get();
//         }

//         // Min and max number of bins with which to FFA transform in order to
//         // cover all trial periods between period_min and period_max.
//         // NOTE: bstop is INclusive
//         // Also, we MUST enforce bstop <= n, to avoid doing an FFA transform with 0 rows
//         const size_t bstart = bins_min;
//         const size_t bstop = std::min({ bins_max, n, size_t(period_max_samples) });

//         /* FFA transform loop */
//         for (size_t bins = bstart; bins <= bstop; ++bins)
//             {
//             const size_t rows = n / bins;
//             const float stdnoise = sqrt(rows * downsampled_variance(size, f));
//             const double period_ceil = std::min(period_max_samples, bins + 1.0);
//             const size_t rows_eval = std::min(rows, ceilshift(rows, bins, period_ceil));

//             transform(input, rows, bins, ffabuf, ffaout);
            
//             auto block = ConstBlock(ffaout, rows_eval, bins);
//             snr2(block, widths, num_widths, stdnoise, snr);

//             for (size_t s = 0; s < rows_eval; ++s)
//                 {
//                 periods[s] = tau * bins * bins / (bins - s / (rows - 1.0));
//                 foldbins[s] = bins;
//                 }

//             snr += rows_eval * num_widths;
//             periods += rows_eval;
//             foldbins += rows_eval;
//             }
//         }
    }

void vis_ffa_transform(
    const std::complex<float>* __restrict__ vis_data,
    size_t size,
    double tsamp,
    double period_min,
    double period_max,
    size_t bins_min,
    size_t bins_max,
    std::vector<std::vector<double>>& block_periods,
    std::vector<std::vector<uint32_t>>& block_foldbins,
    double* __restrict__ base_periods,
    double* __restrict__ tsamps,
    std::vector<ConstComplexBlock>& blocks,
    std::vector<std::unique_ptr<std::complex<float>[]>>& owned_blocks
    )
    {
    periodogram_check_arguments(size, tsamp, period_min, period_max, bins_min, bins_max);

    // Initial downsampling factor
    // We want: ds_ini * tsamp * bmin = period_min
    double ds_ini = period_min / (tsamp * bins_min);

    // Geometric growth factor for the downsampling factor
    double ds_geo = (bins_max + 1.0) / bins_min;

    // Number of required downsampling cycles
    size_t num_downsamplings = ceil(log(period_max / period_min) / log(ds_geo));

    // Allocate buffers
    const size_t complex_bufsize = downsampled_size(size, ds_ini);
    std::unique_ptr<std::complex<float>[]> input_mem(new std::complex<float>[complex_bufsize]);
    std::unique_ptr<std::complex<float>[]> ffabuf_mem(new std::complex<float>[complex_bufsize]);
    std::unique_ptr<std::complex<float>[]> ffaout_mem(new std::complex<float>[complex_bufsize]);
    const std::complex<float>* input = input_mem.get();
    //std::complex<float>* input = input_mem.get();
    std::complex<float>* ffabuf = ffabuf_mem.get();
    std::complex<float>* ffaout = ffaout_mem.get();

    /* Downsampling loop */
    for (size_t ids = 0; ids < num_downsamplings; ++ids)
        {
        const double f = ds_ini * pow(ds_geo, ids); // current downsampling factor
        const double tau = f * tsamp; // current sampling time
        const double period_max_samples = period_max / tau;
        const size_t n = downsampled_size(size, f); // current number of input samples
        
        // downsample() requires f > 1, but we still allow searching the data at their
        // original resolution.
        if (f == 1) {
            input = vis_data;
        }            
        else {
            complex_downsample(vis_data, size, f, input_mem.get());
            input = input_mem.get();
        }

        // Min and max number of bins with which to FFA transform in order to
        // cover all trial periods between period_min and period_max.
        // NOTE: bstop is INclusive
        // Also, we MUST enforce bstop <= n, to avoid doing an FFA transform with 0 rows
        const size_t bstart = bins_min;
        const size_t bstop = std::min({ bins_max, n, size_t(period_max_samples) });
        size_t i = 0;
        /* FFA transform loop */
        for (size_t bins = bstart; bins <= bstop; ++bins)
            {
            const size_t rows = n / bins;
            const double period_ceil = std::min(period_max_samples, bins + 1.0);
            const size_t rows_eval = std::min(rows, ceilshift(rows, bins, period_ceil));
            
            transform(input, rows, bins, ffabuf, ffaout);

            std::vector<double> this_block_periods(rows_eval);
            std::vector<uint32_t> this_block_foldbins(rows_eval);

            for (size_t s = 0; s < rows_eval; ++s) {
                this_block_periods[s] = tau * bins * bins / (bins - s / (rows - 1.0));
                this_block_foldbins[s] = bins;
            }
            
            // copy data pointed to by ffaout to a new block of memory
            std::unique_ptr<std::complex<float>[]> block_mem(new std::complex<float>[rows_eval * bins]);
            std::copy(ffaout, ffaout + (rows_eval * bins), block_mem.get());
            blocks.emplace_back(block_mem.get(), rows_eval, bins);
            // keep track of memory
            owned_blocks.emplace_back(std::move(block_mem));
            block_periods.emplace_back(std::move(this_block_periods));
            block_foldbins.emplace_back(std::move(this_block_foldbins));

            base_periods[i] = tau * bins;
            tsamps[i] = tau;
            ++i;

            }
        }
    }

/**
 * Perform FFA transform on all UV-cell time series at a given base period (bins).
 *
 * @param vis_data   Pointer to dense array of shape (N_uv, nsamp) stored row-major.
 *                   Each row is a time series for one UV cell.
 * @param N_uv       Number of UV cells (first dimension).
 * @param nsamp      Number of time samples (second dimension).
 * @param bins       Number of bins (base period folding factor).
 * @param out        Output buffer (N_uv * rows * bins). Caller allocates.
 *                   The output for UV cell i is stored at offset i * (rows * bins).
 */
void vis_ffa_transform_basep(
    const std::complex<float>* __restrict__ vis_data,
    size_t N_uv,
    size_t nsamp,
    size_t bins,
    double tsamp,
    std::complex<float>* __restrict__ out,
    std::vector<double>& periods_out
) {
    // Number of rows in folding matrix
    const size_t rows = nsamp / bins;
    if (rows < 2) {
        throw std::runtime_error("vis_ffa_transform_basep: not enough rows for given bins.");
    }

    // Compute periods vector (same for all uv-cells)
    periods_out.resize(rows);
    double tau = tsamp;  // current sampling time (no downsampling in this basep mode)
    for (size_t s = 0; s < rows; ++s) {
        periods_out[s] = tau * bins * bins / (bins - s / (rows - 1.0));
    }

    // Temporary buffers for single-series transform
    std::vector<std::complex<float>> ffabuf(rows * bins);
    std::vector<std::complex<float>> ffaout(rows * bins);

    // Iterate over all UV cells
    for (size_t iuv = 0; iuv < N_uv; ++iuv) {
        // Pointer to this UV cell's time series
        const std::complex<float>* ts = vis_data + iuv * nsamp;

        // Run FFA transform for this UV cell
        transform(ts, rows, bins, ffabuf.data(), ffaout.data());

        // Copy result into output cube at correct offset
        std::complex<float>* out_ptr = out + iuv * rows * bins;
        std::copy(ffaout.begin(), ffaout.end(), out_ptr);
    }
}

#endif // PERIODOGRAM_HPP