from cupyx.time import repeat
import pyfftw
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from cupy.cuda import Device
from cupy.fft.config import get_plan_cache
from cupyx.scipy.fft import get_fft_plan


def create_array(*size):
    arr = pyfftw.empty_aligned(size, dtype='complex64')
    arr[:] = np.random.randn(*size) + 1j*np.random.randn(*size)
    return arr


# small = create_array(8, 512, 512)
# big = create_array(8, 2048, 2048)
small = np.empty((8, 512, 512), dtype=np.complex64)
big = np.empty((8, 1024, 1024), dtype=np.complex64)
small[:] = np.random.randn(8, 512, 512) + 1j*np.random.randn(8, 512, 512)
big[:] = np.random.randn(8, 1024, 1024) + 1j*np.random.randn(8, 1024, 1024)

small_c = cp.asarray(small)
big_c = cp.asarray(big)
plan_fft_s = get_fft_plan(small_c, axes=(1, 2), value_type="C2C")
plan_fft_b = get_fft_plan(big_c, axes=(1, 2), value_type="C2C")


def cuda_fft(a):
    b = cp.fft.fftn(a)
    c = cp.fft.ifftn(b)

def plan_small(a):
    plan_fft_s.fft(a, a, cp.cuda.cufft.CUFFT_FORWARD)
    plan_fft_s.fft(a, a, cp.cuda.cufft.CUFFT_INVERSE)

def plan_big(a):
    plan_fft_b.fft(a, a, cp.cuda.cufft.CUFFT_FORWARD)
    plan_fft_b.fft(a, a, cp.cuda.cufft.CUFFT_INVERSE)

Nit = 200
with Device(0):
    cache = get_plan_cache()
    cache.clear()
  # cache.set_size(0)  # disable the cache
#   print("Initial cache  ", cache.get_curr_size())
#   times_small_p = repeat('plan_small(small_c)', repeat=Nit, number=1, globals=globals())
#   print("Run 0 with plan ", cache.get_curr_size())
#   times_big_p = repeat('plan_big(big_c)', repeat=Nit, number=1, globals=globals())
#   print("Run 1 with plan ", cache.get_curr_size())
#   times_small = repeat('cuda_fft(small_c)', repeat=Nit, number=1, globals=globals())
#   print("Run 2 without plan ", cache.get_curr_size())
#   times_big = repeat('cuda_fft(big_c)', repeat=Nit, number=1, globals=globals())
#   print("Run 3 without plan ", cache.get_curr_size())
    # # with plan first
    # print("Initial cache  ", cache.get_curr_size())
    # print(repeat(plan_small, (small_c,), n_repeat=Nit))
    # print("Run 0 with plan ", cache.get_curr_size())
    # print(repeat(plan_big, (big_c,), n_repeat=Nit))
    # print("Run 1 with plan ", cache.get_curr_size())
    # print(repeat(cuda_fft, (small_c,), n_repeat=Nit))
    # print("Run 2 without plan ", cache.get_curr_size())
    # print(repeat(cuda_fft, (big_c,), n_repeat=Nit))
    # print("Run 3 without plan ", cache.get_curr_size())
    # without plan first
    print("Initial cache  ", cache.get_curr_size())
    t_small = repeat(cuda_fft, (small_c,), n_repeat=Nit)
    times_small = t_small.gpu_times[0, :]
    times_small_cpu = t_small.cpu_times
    print("Run 0 without plan ", cache.get_curr_size())
    t_big = repeat(cuda_fft, (big_c,), n_repeat=Nit)
    times_big = t_big.gpu_times[0, :]
    times_big_cpu = t_big.cpu_times
    print("Run 1 without plan ", cache.get_curr_size())
    t_small_p = repeat(plan_small, (small_c,), n_repeat=Nit)
    times_small_p = t_small_p.gpu_times[0, :]
    times_small_p_cpu = t_small_p.cpu_times
    print("Run 2 with plan ", cache.get_curr_size())
    t_big_p = repeat(plan_big, (big_c,), n_repeat=Nit)
    times_big_p = t_big_p.gpu_times[0, :]
    times_big_p_cpu = t_big_p.cpu_times
    print("Run 3 with plan ", cache.get_curr_size())
fig1, ax1 = plt.subplots(1, 2)
ax1[0].plot(times_small)
ax1[0].plot(times_big)
ax1[0].plot(times_small_p)
ax1[0].plot(times_big_p)
ax1[0].set_xlabel("Iteration")
ax1[0].set_ylabel("Time in s")
ax1[0].legend([f"Without plan {small_c.shape}", f"Without plan {big_c.shape}", f"With plan {small_c.shape}", f"With plan {big_c.shape}"])
ax1[0].set_yscale("log")
ax1[0].set_title("GPU Time per iteration")
ax1[1].plot(times_small_cpu)
ax1[1].plot(times_big_cpu)
ax1[1].plot(times_small_p_cpu)
ax1[1].plot(times_big_p_cpu)
ax1[1].set_xlabel("Iteration")
ax1[1].set_ylabel("Time in s")
ax1[1].legend([f"Without plan {small_c.shape}", f"Without plan {big_c.shape}", f"With plan {small_c.shape}", f"With plan {big_c.shape}"])
ax1[1].set_yscale("log")
ax1[1].set_title("CPU Time per iteration")
fig1.suptitle("With plan first")
plt.show()

