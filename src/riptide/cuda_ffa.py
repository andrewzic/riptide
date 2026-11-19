"""
Pure-Python CUDA Fast Folding Algorithm (FFA) runtime backend using CuPy NVRTC.

Provides exact bit-for-bit parity with the C++ ``riptide`` FFA tree split,
``float32`` butterfly shift arithmetic, butterfly rotation sign, and
``complex_downsample`` semantics without requiring an ``nvcc`` build.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cupy as cp

    _CUPY_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:
    cp = None  # type: ignore[assignment]
    _CUPY_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)

_BLOCK_THREADS = 256
_WARPS_PER_BLOCK = _BLOCK_THREADS // 32
_MAX_GRID_Y = 65535
_MAX_PLAN_CACHE = 512

_CUDA_KERNEL_SRC = r"""
extern "C" __global__ void csr_to_uniform_dga_scatter_kernel(
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

extern "C" __global__ void complex_downsample_cuda_kernel(
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

extern "C" __global__ void fold_init_soa_cuda_kernel(
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

extern "C" __global__ void merge_butterfly_cuda_kernel(
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

extern "C" __global__ void pack_soa_to_complex_cuda_kernel(
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
"""

_MODULE_CACHE: Dict[int, Any] = {}
_DEVICE_PLAN_CACHE: "OrderedDict[Tuple[int, int, int], List[Tuple[Any, Any, Any]]]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


class _FFANode:
    """Recursive node describing the unpadded ``m >> 1`` FFA tree decomposition."""

    __slots__ = ("offset", "m", "height", "left", "right")

    def __init__(self, offset: int, m: int) -> None:
        self.offset = offset
        self.m = m
        self.height = 0
        self.left: Optional[_FFANode] = None
        self.right: Optional[_FFANode] = None


def is_cuda_available() -> bool:
    """Return True if CuPy is importable and at least one CUDA device is accessible."""
    if cp is None:
        logger.debug("CuPy unavailable for cuda_ffa: %s", _CUPY_IMPORT_ERROR)
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except cp.cuda.runtime.CUDARuntimeError as exc:
        logger.warning("CUDA device query failed in cuda_ffa: %s", exc)
        return False


def _stream_context(stream: Optional[Any]) -> Any:
    """Normalise an optional CuPy stream or raw stream pointer into a context manager."""
    if stream is None:
        return contextlib.nullcontext()
    if hasattr(stream, "__enter__") and hasattr(stream, "__exit__"):
        return stream
    if isinstance(stream, int):
        return cp.cuda.Stream.null if stream == 0 else cp.cuda.ExternalStream(stream)
    raise TypeError(f"Unsupported CUDA stream type: {type(stream).__name__}")


def _require_cuda() -> None:
    """Raise an explicit RuntimeError if CuPy or a CUDA device is unavailable."""
    if cp is None:
        raise RuntimeError(
            f"CuPy is required for CUDA FFA execution, but failed to import: {_CUPY_IMPORT_ERROR}"
        )
    try:
        n_dev = int(cp.cuda.runtime.getDeviceCount())
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError(f"CUDA device query failed: {exc}") from exc
    if n_dev <= 0:
        raise RuntimeError("CUDA FFA execution requested, but no CUDA devices were detected.")


def _get_module() -> Any:
    """Lazily compile and cache the NVRTC RawModule for the active CUDA device."""
    _require_cuda()
    dev_id = int(cp.cuda.Device().id)
    with _CACHE_LOCK:
        mod = _MODULE_CACHE.get(dev_id)
        if mod is None:
            mod = cp.RawModule(
                code=_CUDA_KERNEL_SRC,
                options=(
                    "--std=c++17",
                    "--generate-code=arch=compute_75,code=sm_75",
                    "--generate-code=arch=compute_80,code=sm_80",
                    "--generate-code=arch=compute_80,code=compute_80",
                ),
            )
            _MODULE_CACHE[dev_id] = mod
        return mod


def _build_ffa_tree(offset: int, m: int) -> _FFANode:
    node = _FFANode(offset, m)
    if m <= 1:
        node.height = 0
        return node
    m_top = m >> 1
    m_bot = m - m_top
    node.left = _build_ffa_tree(offset, m_top)
    node.right = _build_ffa_tree(offset + m_top, m_bot)
    node.height = 1 + max(node.left.height, node.right.height)
    return node


def _emit_ffa_level(
    node: _FFANode,
    target_height: int,
    bins: int,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    shift: np.ndarray,
) -> None:
    if node.height < target_height:
        sl = slice(node.offset, node.offset + node.m)
        idx_a[sl] = np.arange(node.offset, node.offset + node.m, dtype=np.int32)
        idx_b[sl] = -1
        shift[sl] = 0
        return

    if node.height == target_height:
        assert node.left is not None and node.right is not None
        m = node.m
        m_top = node.left.m
        m_bot = node.right.m
        off_top = node.left.offset
        off_bot = node.right.offset

        one_f32 = np.float32(1.0)
        half_f32 = np.float32(0.5)
        kh = np.float32((np.float32(m_top) - one_f32) / (np.float32(m) - one_f32))
        kt = np.float32((np.float32(m_bot) - one_f32) / (np.float32(m) - one_f32))

        s_arr = np.arange(m, dtype=np.int64)
        s_f32 = s_arr.astype(np.float32)

        h = (kh * s_f32 + half_f32).astype(np.int64)
        t = (kt * s_f32 + half_f32).astype(np.int64)
        b = s_arr - (h + t)
        r = (h + b) % bins

        sl = slice(node.offset, node.offset + m)
        idx_a[sl] = (off_top + h).astype(np.int32)
        idx_b[sl] = (off_bot + t).astype(np.int32)
        shift[sl] = r.astype(np.int32)
        return

    assert node.left is not None and node.right is not None
    _emit_ffa_level(node.left, target_height, bins, idx_a, idx_b, shift)
    _emit_ffa_level(node.right, target_height, bins, idx_a, idx_b, shift)


def plan_ffa_levels(rows: int, bins: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build bottom-up level-synchronous butterfly schedules ``(idx_a, idx_b, shift)``
    matching C++ ``riptide::transform`` for arbitrary ``rows >= 2``.

    Rows belonging to shallower already-completed subtrees receive ``idx_b = -1``
    so they copy cleanly across ping-pong buffers.
    """
    rows = int(rows)
    bins = int(bins)
    if rows < 2:
        raise ValueError(f"plan_ffa_levels requires rows >= 2; got {rows}.")
    if bins < 1:
        raise ValueError(f"plan_ffa_levels requires bins >= 1; got {bins}.")

    tree = _build_ffa_tree(0, rows)
    levels: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for target_height in range(1, tree.height + 1):
        idx_a = np.empty(rows, dtype=np.int32)
        idx_b = np.empty(rows, dtype=np.int32)
        shift = np.empty(rows, dtype=np.int32)
        _emit_ffa_level(tree, target_height, bins, idx_a, idx_b, shift)
        levels.append((idx_a, idx_b, shift))
    return levels


def _get_device_plan(rows: int, bins: int) -> List[Tuple[Any, Any, Any]]:
    """Return cached device ``(d_idx_a, d_idx_b, d_shift)`` schedules for ``(device, rows, bins)``."""
    _require_cuda()
    dev_id = int(cp.cuda.Device().id)
    key = (dev_id, int(rows), int(bins))
    with _CACHE_LOCK:
        cached = _DEVICE_PLAN_CACHE.get(key)
        if cached is not None:
            _DEVICE_PLAN_CACHE.move_to_end(key)
            return cached

    host_levels = plan_ffa_levels(rows, bins)
    dev_levels = [
        (cp.asarray(ia), cp.asarray(ib), cp.asarray(sh))
        for ia, ib, sh in host_levels
    ]
    # Synchronise current stream before exposing cached arrays across streams.
    cp.cuda.get_current_stream().synchronize()

    with _CACHE_LOCK:
        existing = _DEVICE_PLAN_CACHE.get(key)
        if existing is not None:
            _DEVICE_PLAN_CACHE.move_to_end(key)
            return existing
        _DEVICE_PLAN_CACHE[key] = dev_levels
        _DEVICE_PLAN_CACHE.move_to_end(key)
        while len(_DEVICE_PLAN_CACHE) > _MAX_PLAN_CACHE:
            _DEVICE_PLAN_CACHE.popitem(last=False)
    return dev_levels


def csr_to_uniform_dga_scatter_gpu(
    cell_offsets: Any,
    vis_data: Any,
    vis_times: Any,
    nsamp: int,
    dt: float,
    t0: float = 0.0,
    return_device: bool = False,
    stream: Optional[Any] = None,
) -> Any:
    """
    Scatter a CSR visibility store onto a uniform ``(M, nsamp)`` ``complex64``
    Dynamic Grid Array on CUDA.
    """
    _require_cuda()
    dt = float(dt)
    t0 = float(t0)
    nsamp = int(nsamp)
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0; got {dt}")
    if nsamp < 0:
        raise ValueError(f"nsamp must be >= 0; got {nsamp}")

    with _stream_context(stream):
        d_offsets = cp.ascontiguousarray(cp.asarray(cell_offsets), dtype=cp.uint64)
        d_vis = cp.ascontiguousarray(cp.asarray(vis_data), dtype=cp.complex64)
        d_times = cp.ascontiguousarray(cp.asarray(vis_times), dtype=cp.float64)

        if d_offsets.ndim != 1 or int(d_offsets.size) < 1:
            raise ValueError("cell_offsets must be a 1D array of length M + 1 >= 1.")
        if d_vis.ndim != 1 or d_times.ndim != 1 or int(d_vis.size) != int(d_times.size):
            raise ValueError("vis_data and vis_times must be 1D arrays of equal length.")

        M = int(d_offsets.size) - 1
        nnz = int(d_vis.size)
        d_dga = cp.zeros((M, nsamp), dtype=cp.complex64)

        if M > 0 and nsamp > 0 and nnz > 0:
            mod = _get_module()
            scatter_kernel = mod.get_function("csr_to_uniform_dga_scatter_kernel")
            blocks = (M + _WARPS_PER_BLOCK - 1) // _WARPS_PER_BLOCK
            scatter_kernel(
                (blocks,),
                (_BLOCK_THREADS,),
                (
                    d_offsets,
                    d_vis,
                    d_times,
                    d_dga,
                    np.uint64(M),
                    np.uint64(nnz),
                    np.uint64(nsamp),
                    np.float64(dt),
                    np.float64(t0),
                ),
            )

        return d_dga if return_device else cp.asnumpy(d_dga)


def cu_vis_ffa_transform_basep_gpu(
    vis_data: Any,
    bins: int,
    tsamp: float,
    rows_eval: int = 0,
    downsample_factor: float = 1.0,
    return_device: bool = False,
    stream: Optional[Any] = None,
) -> Tuple[Any, np.ndarray]:
    """
    Compute the base-period FFA transform of ``vis_data`` ``(N_uv, nsamp)`` on CUDA.

    Parameters
    ----------
    vis_data : array_like (NumPy or CuPy)
        Complex64 2D array of shape ``(N_uv, nsamp)``.
    bins : int
        Number of phase bins (base period in samples).
    tsamp : float
        Effective sampling interval ``tau = downsample_factor * native_tsamp``.
    rows_eval : int, optional
        Number of shift rows to retain (0 means all rows).
    downsample_factor : float, optional
        Downsampling factor ``>= 1.0`` applied prior to folding.
    return_device : bool, optional
        If True, return the folded output cube as a CuPy array on GPU.
    stream : cupy.cuda.Stream or int, optional
        Optional CUDA stream on which to enqueue operations.

    Returns
    -------
    ffa_out : np.ndarray or cupy.ndarray
        Complex64 array of shape ``(N_uv, actual_rows_eval, bins)``.
    periods : np.ndarray
        Float64 trial periods of shape ``(actual_rows_eval,)``.
    """
    _require_cuda()
    bins = int(bins)
    tsamp = float(tsamp)
    rows_eval = int(rows_eval)
    downsample_factor = float(downsample_factor)

    if bins < 1:
        raise ValueError(f"bins must be >= 1; got {bins}")
    if downsample_factor < 1.0:
        raise ValueError(f"downsample_factor must be >= 1.0; got {downsample_factor}")

    with _stream_context(stream):
        d_vis = cp.ascontiguousarray(cp.asarray(vis_data), dtype=cp.complex64)
        if d_vis.ndim != 2:
            raise RuntimeError(
                f"cu_vis_ffa_transform_basep_gpu: input must be 2D (N_uv, nsamp); got {d_vis.shape}"
            )

        N_uv = int(d_vis.shape[0])
        nsamp = int(d_vis.shape[1])

        do_downsample = downsample_factor > 1.0
        if do_downsample and downsample_factor > float(nsamp):
            raise ValueError("Downsampling factor must verify: 1 < f <= nsamp")

        n = int(math.floor(nsamp / downsample_factor)) if do_downsample else nsamp
        rows = n // bins
        if rows < 2:
            raise RuntimeError("cu_vis_ffa_transform_basep_gpu: not enough rows for given bins.")

        actual_rows_eval = rows_eval if (0 < rows_eval < rows) else rows
        s_idx = np.arange(actual_rows_eval, dtype=np.float64)
        periods = tsamp * bins * bins / (bins - s_idx / (rows - 1.0))

        if N_uv == 0:
            d_empty = cp.empty((0, actual_rows_eval, bins), dtype=cp.complex64)
            return (d_empty if return_device else cp.asnumpy(d_empty)), periods

        mod = _get_module()
        grid_y = min(N_uv, _MAX_GRID_Y)

        if do_downsample:
            d_ds = cp.empty((N_uv, n), dtype=cp.complex64)
            ds_kernel = mod.get_function("complex_downsample_cuda_kernel")
            grid_ds = ((n + _BLOCK_THREADS - 1) // _BLOCK_THREADS, grid_y, 1)
            ds_kernel(
                grid_ds,
                (_BLOCK_THREADS,),
                (d_vis, d_ds, np.uint64(N_uv), np.uint64(nsamp), np.uint64(n), np.float64(downsample_factor)),
            )
            d_fold_src = d_ds
            n_in = n
        else:
            d_ds = None
            d_fold_src = d_vis
            n_in = nsamp

        dev_levels = _get_device_plan(rows, bins)
        num_levels = len(dev_levels)

        d_a_re = cp.empty((N_uv, rows, bins), dtype=cp.float32)
        d_a_im = cp.empty((N_uv, rows, bins), dtype=cp.float32)
        d_b_re = cp.empty((N_uv, rows, bins), dtype=cp.float32)
        d_b_im = cp.empty((N_uv, rows, bins), dtype=cp.float32)

        fold_kernel = mod.get_function("fold_init_soa_cuda_kernel")
        rb = rows * bins
        fold_kernel(
            ((rb + _BLOCK_THREADS - 1) // _BLOCK_THREADS, grid_y, 1),
            (_BLOCK_THREADS,),
            (d_fold_src, d_a_re, d_a_im, np.uint64(N_uv), np.uint64(n_in), np.uint64(rows), np.uint64(bins)),
        )
        del d_fold_src, d_ds

        merge_kernel = mod.get_function("merge_butterfly_cuda_kernel")
        src_re, src_im = d_a_re, d_a_im
        dst_re, dst_im = d_b_re, d_b_im

        for lvl, (d_ia, d_ib, d_sh) in enumerate(dev_levels):
            eval_rows = actual_rows_eval if (lvl == num_levels - 1) else rows
            eval_rb = eval_rows * bins
            merge_kernel(
                ((eval_rb + _BLOCK_THREADS - 1) // _BLOCK_THREADS, grid_y, 1),
                (_BLOCK_THREADS,),
                (
                    src_re,
                    src_im,
                    dst_re,
                    dst_im,
                    d_ia,
                    d_ib,
                    d_sh,
                    np.uint64(N_uv),
                    np.uint64(rows),
                    np.uint64(eval_rows),
                    np.uint64(bins),
                ),
            )
            src_re, dst_re = dst_re, src_re
            src_im, dst_im = dst_im, src_im

        # Release the unused ping-pong pair before allocating output so peak scratch is 4 float32 planes.
        del dst_re, dst_im, d_a_re, d_a_im, d_b_re, d_b_im

        d_out = cp.empty((N_uv, actual_rows_eval, bins), dtype=cp.complex64)
        pack_kernel = mod.get_function("pack_soa_to_complex_cuda_kernel")
        out_rb = actual_rows_eval * bins
        pack_kernel(
            ((out_rb + _BLOCK_THREADS - 1) // _BLOCK_THREADS, grid_y, 1),
            (_BLOCK_THREADS,),
            (
                src_re,
                src_im,
                d_out,
                np.uint64(N_uv),
                np.uint64(rows),
                np.uint64(actual_rows_eval),
                np.uint64(bins),
            ),
        )
        del src_re, src_im

        return (d_out if return_device else cp.asnumpy(d_out)), periods


__all__ = [
    "is_cuda_available",
    "plan_ffa_levels",
    "csr_to_uniform_dga_scatter_gpu",
    "cu_vis_ffa_transform_basep_gpu",
]
