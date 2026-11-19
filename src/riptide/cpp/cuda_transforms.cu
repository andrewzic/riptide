#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define RIPTIDE_CUDA_CHECK(call)                                              \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            throw std::runtime_error(                                         \
                std::string("CUDA error at ") + __FILE__ + ":" +              \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err__)); \
        }                                                                     \
    } while (0)

#define RIPTIDE_CUDA_CHECK_LAST()                                             \
    do {                                                                      \
        cudaError_t err__ = cudaGetLastError();                               \
        if (err__ != cudaSuccess) {                                           \
            throw std::runtime_error(                                         \
                std::string("CUDA kernel launch error at ") + __FILE__ + ":" +\
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err__)); \
        }                                                                     \
    } while (0)

namespace riptide {
namespace cuda_detail {

constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize;
constexpr size_t kMaxGridY = 65535;

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) : count(n) {
        if (n > 0) {
            RIPTIDE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
        }
    }
    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), count(other.count) {
        other.ptr = nullptr;
        other.count = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
            ptr = other.ptr;
            count = other.count;
            other.ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }
};

template <typename T>
struct StreamBuffer {
    T* ptr = nullptr;
    size_t count = 0;
    cudaStream_t stream = nullptr;
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
    bool is_async = false;
#endif

    StreamBuffer() = default;
    StreamBuffer(size_t n, cudaStream_t s) : count(n), stream(s) {
        if (n == 0) {
            return;
        }
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
        int dev = 0;
        int mem_pools_supported = 0;
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&mem_pools_supported, cudaDevAttrMemoryPoolsSupported, dev) == cudaSuccess &&
            mem_pools_supported != 0) {
            RIPTIDE_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ptr), n * sizeof(T), stream));
            is_async = true;
            return;
        }
#endif
        RIPTIDE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
    }

    ~StreamBuffer() {
        release();
    }

    void release() noexcept {
        if (ptr != nullptr) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
            if (is_async) {
                cudaFreeAsync(ptr, stream);
            } else {
                cudaFree(ptr);
            }
#else
            cudaFree(ptr);
#endif
            ptr = nullptr;
            count = 0;
        }
    }

    StreamBuffer(const StreamBuffer&) = delete;
    StreamBuffer& operator=(const StreamBuffer&) = delete;

    StreamBuffer(StreamBuffer&& other) noexcept
        : ptr(other.ptr),
          count(other.count),
          stream(other.stream)
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
          , is_async(other.is_async)
#endif
    {
        other.ptr = nullptr;
        other.count = 0;
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
        other.is_async = false;
#endif
    }

    StreamBuffer& operator=(StreamBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr = other.ptr;
            count = other.count;
            stream = other.stream;
            other.ptr = nullptr;
            other.count = 0;
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
            is_async = other.is_async;
            other.is_async = false;
#endif
        }
        return *this;
    }
};

struct FFANode {
    size_t offset = 0;
    size_t m = 0;
    int height = 0;
    std::unique_ptr<FFANode> left;
    std::unique_ptr<FFANode> right;
};

inline std::unique_ptr<FFANode> build_ffa_tree(size_t offset, size_t m) {
    auto node = std::make_unique<FFANode>();
    node->offset = offset;
    node->m = m;
    if (m <= 1) {
        node->height = 0;
        return node;
    }
    const size_t m_top = m >> 1;
    const size_t m_bot = m - m_top;
    node->left = build_ffa_tree(offset, m_top);
    node->right = build_ffa_tree(offset + m_top, m_bot);
    node->height = 1 + std::max(node->left->height, node->right->height);
    return node;
}

inline void emit_ffa_level(
    const FFANode* node,
    int target_height,
    size_t bins,
    int* idx_a,
    int* idx_b,
    int* shift
) {
    if (node->height < target_height) {
        // Subtree finalised at an earlier level (or leaf m==1); copy through unchanged.
        for (size_t s = 0; s < node->m; ++s) {
            const size_t row = node->offset + s;
            idx_a[row] = static_cast<int>(row);
            idx_b[row] = -1;
            shift[row] = 0;
        }
        return;
    }

    if (node->height == target_height) {
        const size_t m = node->m;
        const size_t m_top = node->left->m;
        const size_t m_bot = node->right->m;
        const size_t off_top = node->left->offset;
        const size_t off_bot = node->right->offset;

        const float kh = (m_top - 1.0f) / (m - 1.0f);
        const float kt = (m_bot - 1.0f) / (m - 1.0f);

        for (size_t s = 0; s < m; ++s) {
            const size_t h = static_cast<size_t>(kh * s + 0.5f);
            const size_t t = static_cast<size_t>(kt * s + 0.5f);
            const size_t b = s - (h + t);
            const size_t r = (h + b) % bins;

            const size_t row = node->offset + s;
            idx_a[row] = static_cast<int>(off_top + h);
            idx_b[row] = static_cast<int>(off_bot + t);
            shift[row] = static_cast<int>(r);
        }
        return;
    }

    emit_ffa_level(node->left.get(), target_height, bins, idx_a, idx_b, shift);
    emit_ffa_level(node->right.get(), target_height, bins, idx_a, idx_b, shift);
}

constexpr size_t kMaxPlanCache = 512;

struct PlanKey {
    int device_id = 0;
    size_t rows = 0;
    size_t bins = 0;

    bool operator==(const PlanKey& o) const noexcept {
        return device_id == o.device_id && rows == o.rows && bins == o.bins;
    }
};

struct PlanKeyHash {
    size_t operator()(const PlanKey& k) const noexcept {
        size_t h = std::hash<int>{}(k.device_id);
        h ^= std::hash<size_t>{}(k.rows) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(k.bins) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct CachedFFAPlan {
    int device_id = 0;
    int num_levels = 0;
    size_t rows = 0;
    int* d_sched = nullptr;

    ~CachedFFAPlan() {
        if (d_sched != nullptr) {
            int prev_dev = 0;
            bool switched = false;
            if (cudaGetDevice(&prev_dev) == cudaSuccess && prev_dev != device_id) {
                if (cudaSetDevice(device_id) == cudaSuccess) {
                    switched = true;
                }
            }
            cudaFree(d_sched);
            cudaGetLastError();
            if (switched) {
                cudaSetDevice(prev_dev);
            }
            d_sched = nullptr;
        }
    }
};

class FFAPlanCache {
public:
    static FFAPlanCache& instance() {
        static FFAPlanCache cache;
        return cache;
    }

    std::shared_ptr<const CachedFFAPlan> get_or_create(
        int device_id,
        size_t rows,
        size_t bins,
        cudaStream_t upload_stream
    ) {
        const PlanKey key{device_id, rows, bins};
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = map_.find(key);
            if (it != map_.end()) {
                lru_.splice(lru_.begin(), lru_, it->second);
                return it->second->second;
            }
        }

        auto tree = build_ffa_tree(0, rows);
        const int num_levels = tree->height;
        const size_t level_stride = 3 * rows;
        std::vector<int> host_sched(static_cast<size_t>(num_levels) * level_stride);
        for (int lvl = 0; lvl < num_levels; ++lvl) {
            int* base = host_sched.data() + static_cast<size_t>(lvl) * level_stride;
            emit_ffa_level(tree.get(), lvl + 1, bins, base, base + rows, base + 2 * rows);
        }

        auto plan = std::make_shared<CachedFFAPlan>();
        plan->device_id = device_id;
        plan->num_levels = num_levels;
        plan->rows = rows;
        if (!host_sched.empty()) {
            RIPTIDE_CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&plan->d_sched),
                host_sched.size() * sizeof(int)
            ));
            RIPTIDE_CUDA_CHECK(cudaMemcpyAsync(
                plan->d_sched,
                host_sched.data(),
                host_sched.size() * sizeof(int),
                cudaMemcpyHostToDevice,
                upload_stream
            ));
            // Synchronise upload stream before inserting into cache to avoid cross-stream races.
            RIPTIDE_CUDA_CHECK(cudaStreamSynchronize(upload_stream));
        }

        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second);
            return it->second->second;
        }
        lru_.emplace_front(key, plan);
        map_[key] = lru_.begin();
        while (lru_.size() > kMaxPlanCache) {
            auto last = std::prev(lru_.end());
            map_.erase(last->first);
            lru_.pop_back();
        }
        return plan;
    }

private:
    using ListType = std::list<std::pair<PlanKey, std::shared_ptr<const CachedFFAPlan>>>;
    std::mutex mu_;
    ListType lru_;
    std::unordered_map<PlanKey, ListType::iterator, PlanKeyHash> map_;
};

__global__ void csr_to_uniform_dga_scatter_kernel(
    const unsigned long long* __restrict__ d_cell_offsets,
    const float2* __restrict__ d_vis_data,
    const double* __restrict__ d_vis_times,
    float2* __restrict__ d_dga_out,
    size_t M,
    size_t nnz,
    size_t nsamp,
    double dt,
    double t0
) {
    const size_t warp_in_block = static_cast<size_t>(threadIdx.x >> 5);
    const size_t lane = static_cast<size_t>(threadIdx.x & 31);
    const size_t m = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x >> 5) + warp_in_block;
    if (m >= M) {
        return;
    }

    const unsigned long long beg = d_cell_offsets[m];
    const unsigned long long end_raw = d_cell_offsets[m + 1];
    const unsigned long long end = (end_raw <= static_cast<unsigned long long>(nnz))
                                       ? end_raw
                                       : static_cast<unsigned long long>(nnz);
    if (beg >= end) {
        return;
    }

    const size_t row_base = m * nsamp;
    for (unsigned long long idx = beg + lane; idx < end; idx += 32ULL) {
        const double t_rel = (d_vis_times[idx] - t0) / dt;
        if (isfinite(t_rel) && t_rel >= -0.5 && t_rel < static_cast<double>(nsamp) + 0.5) {
            const long long k = llround(t_rel);
            if (k >= 0 && k < static_cast<long long>(nsamp)) {
                const size_t dst = row_base + static_cast<size_t>(k);
                const float2 v = d_vis_data[idx];
                atomicAdd(&d_dga_out[dst].x, v.x);
                atomicAdd(&d_dga_out[dst].y, v.y);
            }
        }
    }
}

__global__ void complex_downsample_cuda_kernel(
    const float2* __restrict__ d_in,
    float2* __restrict__ d_out,
    size_t N_uv,
    size_t nsamp,
    size_t n,
    double f
) {
    const size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n) {
        return;
    }

    const double start = static_cast<double>(k) * f;
    const double end = start + f;
    const size_t imin = static_cast<size_t>(floor(start));
    const size_t floor_end = static_cast<size_t>(floor(end));
    const size_t imax = (floor_end < (nsamp - 1)) ? floor_end : (nsamp - 1);

    const bool single_bin = (imin == imax);
    const float w_single = single_bin ? static_cast<float>(end - start) : 0.0f;
    const float wmin = single_bin ? 0.0f : static_cast<float>(static_cast<double>(imin + 1) - start);
    const float wmax = single_bin ? 0.0f : static_cast<float>(end - static_cast<double>(imax));

    for (size_t iuv = blockIdx.y; iuv < N_uv; iuv += gridDim.y) {
        const size_t in_base = iuv * nsamp;
        float acc_re, acc_im;
        if (single_bin) {
            const float2 v = d_in[in_base + imin];
            acc_re = w_single * v.x;
            acc_im = w_single * v.y;
        } else {
            const float2 v0 = d_in[in_base + imin];
            acc_re = wmin * v0.x;
            acc_im = wmin * v0.y;
            for (size_t i = imin + 1; i < imax; ++i) {
                const float2 vi = d_in[in_base + i];
                acc_re += vi.x;
                acc_im += vi.y;
            }
            const float2 v1 = d_in[in_base + imax];
            acc_re += wmax * v1.x;
            acc_im += wmax * v1.y;
        }
        d_out[iuv * n + k] = make_float2(acc_re, acc_im);
    }
}

__global__ void fold_init_soa_cuda_kernel(
    const float2* __restrict__ d_in,
    float* __restrict__ d_out_re,
    float* __restrict__ d_out_im,
    size_t N_uv,
    size_t n_in,
    size_t rows,
    size_t bins
) {
    const size_t rb = rows * bins;
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= rb) {
        return;
    }

    for (size_t iuv = blockIdx.y; iuv < N_uv; iuv += gridDim.y) {
        const float2 v = d_in[iuv * n_in + col];
        const size_t dst = iuv * rb + col;
        d_out_re[dst] = v.x;
        d_out_im[dst] = v.y;
    }
}

__global__ void merge_butterfly_cuda_kernel(
    const float* __restrict__ d_in_re,
    const float* __restrict__ d_in_im,
    float* __restrict__ d_out_re,
    float* __restrict__ d_out_im,
    const int* __restrict__ d_idx_a,
    const int* __restrict__ d_idx_b,
    const int* __restrict__ d_shift,
    size_t N_uv,
    size_t rows,
    size_t eval_rows,
    size_t bins
) {
    const size_t eval_rb = eval_rows * bins;
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= eval_rb) {
        return;
    }

    const size_t rb = rows * bins;
    const size_t s = col / bins;
    const size_t j = col - s * bins;
    const int ia = d_idx_a[s];
    const int ib = d_idx_b[s];
    const size_t a_local = static_cast<size_t>(ia) * bins + j;

    if (ib < 0) {
        for (size_t iuv = blockIdx.y; iuv < N_uv; iuv += gridDim.y) {
            const size_t base = iuv * rb;
            d_out_re[base + col] = d_in_re[base + a_local];
            d_out_im[base + col] = d_in_im[base + a_local];
        }
    } else {
        const size_t r = static_cast<size_t>(d_shift[s]);
        const size_t j_rot = (j + r >= bins) ? (j + r - bins) : (j + r);
        const size_t b_local = static_cast<size_t>(ib) * bins + j_rot;
        for (size_t iuv = blockIdx.y; iuv < N_uv; iuv += gridDim.y) {
            const size_t base = iuv * rb;
            d_out_re[base + col] = d_in_re[base + a_local] + d_in_re[base + b_local];
            d_out_im[base + col] = d_in_im[base + a_local] + d_in_im[base + b_local];
        }
    }
}

__global__ void pack_soa_to_complex_cuda_kernel(
    const float* __restrict__ d_in_re,
    const float* __restrict__ d_in_im,
    float2* __restrict__ d_out,
    size_t N_uv,
    size_t rows,
    size_t actual_rows_eval,
    size_t bins
) {
    const size_t out_rb = actual_rows_eval * bins;
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= out_rb) {
        return;
    }

    const size_t in_rb = rows * bins;
    for (size_t iuv = blockIdx.y; iuv < N_uv; iuv += gridDim.y) {
        const size_t src = iuv * in_rb + col;
        const size_t dst = iuv * out_rb + col;
        d_out[dst] = make_float2(d_in_re[src], d_in_im[src]);
    }
}

__global__ void compute_periods_cuda_kernel(
    double* __restrict__ d_periods_out,
    size_t actual_rows_eval,
    size_t rows,
    size_t bins,
    double tau
) {
    const size_t s = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s < actual_rows_eval) {
        const double b = static_cast<double>(bins);
        d_periods_out[s] = tau * b * b / (b - static_cast<double>(s) / static_cast<double>(rows - 1));
    }
}

inline void vis_ffa_transform_basep_device_core(
    const float2* d_vis_data,
    size_t N_uv,
    size_t nsamp,
    size_t bins,
    double tsamp,
    float2* d_out,
    double* d_periods_out,
    size_t rows_eval,
    double downsample_factor,
    cudaStream_t stream
) {
    if (bins == 0) {
        throw std::invalid_argument("cu_vis_ffa_transform_basep: bins must be >= 1.");
    }
    if (downsample_factor < 1.0) {
        throw std::invalid_argument("cu_vis_ffa_transform_basep: downsample_factor must be >= 1.0.");
    }

    const bool do_downsample = (downsample_factor > 1.0);
    if (do_downsample && downsample_factor > static_cast<double>(nsamp)) {
        throw std::invalid_argument("cu_vis_ffa_transform_basep: downsample_factor must verify 1 < f <= nsamp.");
    }

    const size_t n = do_downsample
                         ? static_cast<size_t>(std::floor(static_cast<double>(nsamp) / downsample_factor))
                         : nsamp;
    const size_t rows = n / bins;
    if (rows < 2) {
        throw std::runtime_error("cu_vis_ffa_transform_basep: not enough rows for given bins.");
    }

    const size_t actual_rows_eval = (rows_eval > 0 && rows_eval < rows) ? rows_eval : rows;

    if (d_periods_out != nullptr) {
        const unsigned int p_blocks = static_cast<unsigned int>((actual_rows_eval + kBlockThreads - 1) / kBlockThreads);
        compute_periods_cuda_kernel<<<p_blocks, kBlockThreads, 0, stream>>>(
            d_periods_out, actual_rows_eval, rows, bins, tsamp
        );
        RIPTIDE_CUDA_CHECK_LAST();
    }

    if (N_uv == 0) {
        return;
    }
    if (d_vis_data == nullptr || d_out == nullptr) {
        throw std::runtime_error("cu_vis_ffa_transform_basep: null device buffer pointer provided.");
    }

    const unsigned int grid_y = static_cast<unsigned int>(std::min(N_uv, kMaxGridY));

    StreamBuffer<float2> d_ds;
    const float2* d_fold_src = d_vis_data;
    size_t n_in = nsamp;

    if (do_downsample) {
        d_ds = StreamBuffer<float2>(N_uv * n, stream);
        const dim3 grid_ds(static_cast<unsigned int>((n + kBlockThreads - 1) / kBlockThreads), grid_y, 1);
        complex_downsample_cuda_kernel<<<grid_ds, kBlockThreads, 0, stream>>>(
            d_vis_data, d_ds.ptr, N_uv, nsamp, n, downsample_factor
        );
        RIPTIDE_CUDA_CHECK_LAST();
        d_fold_src = d_ds.ptr;
        n_in = n;
    }

    int device_id = 0;
    RIPTIDE_CUDA_CHECK(cudaGetDevice(&device_id));
    auto plan = FFAPlanCache::instance().get_or_create(device_id, rows, bins, stream);
    const int num_levels = plan->num_levels;
    const size_t level_stride = 3 * rows;

    const size_t plane_elems = N_uv * rows * bins;
    StreamBuffer<float> d_a_re(plane_elems, stream);
    StreamBuffer<float> d_a_im(plane_elems, stream);
    StreamBuffer<float> d_b_re(plane_elems, stream);
    StreamBuffer<float> d_b_im(plane_elems, stream);

    const size_t rb = rows * bins;
    const dim3 grid_fold(static_cast<unsigned int>((rb + kBlockThreads - 1) / kBlockThreads), grid_y, 1);
    fold_init_soa_cuda_kernel<<<grid_fold, kBlockThreads, 0, stream>>>(
        d_fold_src, d_a_re.ptr, d_a_im.ptr, N_uv, n_in, rows, bins
    );
    RIPTIDE_CUDA_CHECK_LAST();
    d_ds.release();

    float* src_re = d_a_re.ptr;
    float* src_im = d_a_im.ptr;
    float* dst_re = d_b_re.ptr;
    float* dst_im = d_b_im.ptr;

    for (int lvl = 0; lvl < num_levels; ++lvl) {
        const size_t eval_rows = (lvl == num_levels - 1) ? actual_rows_eval : rows;
        const size_t eval_rb = eval_rows * bins;
        const dim3 grid_merge(static_cast<unsigned int>((eval_rb + kBlockThreads - 1) / kBlockThreads), grid_y, 1);
        const int* d_lvl = plan->d_sched + static_cast<size_t>(lvl) * level_stride;

        merge_butterfly_cuda_kernel<<<grid_merge, kBlockThreads, 0, stream>>>(
            src_re,
            src_im,
            dst_re,
            dst_im,
            d_lvl,
            d_lvl + rows,
            d_lvl + 2 * rows,
            N_uv,
            rows,
            eval_rows,
            bins
        );
        RIPTIDE_CUDA_CHECK_LAST();

        std::swap(src_re, dst_re);
        std::swap(src_im, dst_im);
    }

    if (dst_re == d_a_re.ptr) {
        d_a_re.release();
        d_a_im.release();
    } else {
        d_b_re.release();
        d_b_im.release();
    }

    const size_t out_rb = actual_rows_eval * bins;
    const dim3 grid_pack(static_cast<unsigned int>((out_rb + kBlockThreads - 1) / kBlockThreads), grid_y, 1);
    pack_soa_to_complex_cuda_kernel<<<grid_pack, kBlockThreads, 0, stream>>>(
        src_re, src_im, d_out, N_uv, rows, actual_rows_eval, bins
    );
    RIPTIDE_CUDA_CHECK_LAST();
}

} // namespace cuda_detail

bool cu_has_cuda() {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

void cu_csr_to_uniform_dga_scatter(
    const uint64_t* cell_offsets,
    const std::complex<float>* vis_data,
    const double* vis_times,
    size_t M,
    size_t nnz,
    size_t nsamp,
    double dt,
    double t0,
    std::complex<float>* dga_out
) {
    if (dt <= 0.0) {
        throw std::invalid_argument("cu_csr_to_uniform_dga_scatter: dt must be > 0.");
    }
    if (M == 0 || nsamp == 0) {
        return;
    }
    if (cell_offsets == nullptr || dga_out == nullptr) {
        throw std::runtime_error("cu_csr_to_uniform_dga_scatter: null cell_offsets or dga_out pointer.");
    }
    if (nnz == 0) {
        std::memset(dga_out, 0, M * nsamp * sizeof(std::complex<float>));
        return;
    }
    if (vis_data == nullptr || vis_times == nullptr) {
        throw std::runtime_error("cu_csr_to_uniform_dga_scatter: null vis_data or vis_times pointer.");
    }

    cuda_detail::DeviceBuffer<unsigned long long> d_offsets(M + 1);
    cuda_detail::DeviceBuffer<float2> d_vis(nnz);
    cuda_detail::DeviceBuffer<double> d_times(nnz);
    cuda_detail::DeviceBuffer<float2> d_dga(M * nsamp);

    RIPTIDE_CUDA_CHECK(cudaMemcpy(
        d_offsets.ptr, cell_offsets, (M + 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice
    ));
    RIPTIDE_CUDA_CHECK(cudaMemcpy(
        d_vis.ptr, vis_data, nnz * sizeof(float2), cudaMemcpyHostToDevice
    ));
    RIPTIDE_CUDA_CHECK(cudaMemcpy(
        d_times.ptr, vis_times, nnz * sizeof(double), cudaMemcpyHostToDevice
    ));
    RIPTIDE_CUDA_CHECK(cudaMemset(d_dga.ptr, 0, M * nsamp * sizeof(float2)));

    const unsigned int blocks = static_cast<unsigned int>(
        (M + cuda_detail::kWarpsPerBlock - 1) / cuda_detail::kWarpsPerBlock
    );
    cuda_detail::csr_to_uniform_dga_scatter_kernel<<<blocks, cuda_detail::kBlockThreads>>>(
        d_offsets.ptr, d_vis.ptr, d_times.ptr, d_dga.ptr, M, nnz, nsamp, dt, t0
    );
    RIPTIDE_CUDA_CHECK_LAST();

    RIPTIDE_CUDA_CHECK(cudaMemcpy(
        dga_out, d_dga.ptr, M * nsamp * sizeof(float2), cudaMemcpyDeviceToHost
    ));
}

void cu_vis_ffa_transform_basep_device(
    const std::complex<float>* d_vis_data,
    size_t N_uv,
    size_t nsamp,
    size_t bins,
    double tsamp,
    std::complex<float>* d_out,
    double* d_periods_out,
    size_t rows_eval,
    double downsample_factor,
    void* stream
) {
    if (d_periods_out == nullptr) {
        throw std::runtime_error("cu_vis_ffa_transform_basep_device: d_periods_out pointer must not be null.");
    }
    cuda_detail::vis_ffa_transform_basep_device_core(
        reinterpret_cast<const float2*>(d_vis_data),
        N_uv,
        nsamp,
        bins,
        tsamp,
        reinterpret_cast<float2*>(d_out),
        d_periods_out,
        rows_eval,
        downsample_factor,
        static_cast<cudaStream_t>(stream)
    );
}

void cu_vis_ffa_transform_basep_host(
    const std::complex<float>* vis_data,
    size_t N_uv,
    size_t nsamp,
    size_t bins,
    double tsamp,
    std::complex<float>* out,
    double* periods_out,
    size_t rows_eval,
    double downsample_factor
) {
    if (periods_out == nullptr) {
        throw std::runtime_error("cu_vis_ffa_transform_basep_host: periods_out pointer must not be null.");
    }
    if (bins == 0) {
        throw std::invalid_argument("cu_vis_ffa_transform_basep: bins must be >= 1.");
    }
    if (downsample_factor < 1.0) {
        throw std::invalid_argument("cu_vis_ffa_transform_basep: downsample_factor must be >= 1.0.");
    }

    const bool do_downsample = (downsample_factor > 1.0);
    if (do_downsample && downsample_factor > static_cast<double>(nsamp)) {
        throw std::invalid_argument("cu_vis_ffa_transform_basep: downsample_factor must verify 1 < f <= nsamp.");
    }

    const size_t n = do_downsample
                         ? static_cast<size_t>(std::floor(static_cast<double>(nsamp) / downsample_factor))
                         : nsamp;
    const size_t rows = n / bins;
    if (rows < 2) {
        throw std::runtime_error("cu_vis_ffa_transform_basep: not enough rows for given bins.");
    }
    const size_t actual_rows_eval = (rows_eval > 0 && rows_eval < rows) ? rows_eval : rows;

    for (size_t s = 0; s < actual_rows_eval; ++s) {
        periods_out[s] = tsamp * bins * bins / (bins - s / (rows - 1.0));
    }

    if (N_uv == 0) {
        return;
    }
    if (vis_data == nullptr || out == nullptr) {
        throw std::runtime_error("cu_vis_ffa_transform_basep_host: null vis_data or out pointer.");
    }

    cuda_detail::DeviceBuffer<float2> d_in(N_uv * nsamp);
    cuda_detail::DeviceBuffer<float2> d_out(N_uv * actual_rows_eval * bins);

    RIPTIDE_CUDA_CHECK(cudaMemcpy(
        d_in.ptr, vis_data, N_uv * nsamp * sizeof(float2), cudaMemcpyHostToDevice
    ));

    cuda_detail::vis_ffa_transform_basep_device_core(
        d_in.ptr,
        N_uv,
        nsamp,
        bins,
        tsamp,
        d_out.ptr,
        nullptr,
        rows_eval,
        downsample_factor,
        nullptr
    );

    RIPTIDE_CUDA_CHECK(cudaStreamSynchronize(nullptr));
    RIPTIDE_CUDA_CHECK(cudaMemcpy(
        out, d_out.ptr, N_uv * actual_rows_eval * bins * sizeof(float2), cudaMemcpyDeviceToHost
    ));
}

} // namespace riptide
