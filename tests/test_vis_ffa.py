import numpy as np
import riptide
from riptide import vis_ffa_search
from riptide import TimeSeries, VisTimeSeries, UV_FFA
import time
import matplotlib.pyplot as plt

data = np.random.rand(70000) + 1j*np.random.rand(70000)

tsamp=0.110
period_min=tsamp*8
period_max=300.0
bins_min=6
bins_max=256
dts = []
for i in range(100):
    ts = TimeSeries(data, tsamp, dtype=np.complex64)
    uv_ind = (10,20)
    t=time.perf_counter()
    ts, result=vis_ffa_search(ts, uv_ind, period_min=period_min, period_max=period_max, bins_min=bins_min, bins_max=bins_max)
    #result=riptide.libcpp.vis_ffa_transform(data, tsamp, period_min, period_max, bins_min, bins_max)
    t2=time.perf_counter()
    dt=t2-t
    dts.append(dt)
print(f"avg time elapsed: {np.mean(dts):.3f}")

plt.imshow(np.abs(result[4][0]), origin="lower", aspect="auto")
plt.show()
