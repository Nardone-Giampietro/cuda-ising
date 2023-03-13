import logging
import sys
import numpy as np
from pycuda.compiler import SourceModule
from pycuda import autoinit
from pycuda import gpuarray
from pycuda.reduction import ReductionKernel
import matplotlib.pyplot as plt


class DataAnalysis:
    def __init__(self, data, name):

        try:
            iterator = iter(data)
        except TypeError:
            logging.exception(
                f"Argument 'data' of {type(self).__name__} is not iterable."
            )
            sys.exit()

        try:
            if len(data) == 0:
                raise ValueError
        except ValueError:
            logging.exception(f"Imput data of {type(self).__name__} is empty.")
            sys.exit()

        self._data = np.array(data, dtype=np.float32)
        self._len = len(data)
        self._name = str(name)

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._name

    def plot(self, scalex=True, scaley=True, data=None, **kwargs):
        plt.xlabel("Time t")
        plt.ylabel(self._name)
        times = np.arange(self._len, dtype=np.int32)
        plt.plot(
            times, self._data, "ro", scalex=scalex, scaley=scaley, data=data, **kwargs
        )
        plt.show()

    def plot_autocorrelation(
        self, max_distance=500, scalex=True, scaley=True, data=None, **kwargs
    ):
        data_gpu = gpuarray.to_gpu(self._data)
        autocorrelations = gpuarray.zeros(max_distance + 1, dtype=np.float32)
        total = gpuarray.zeros(1, dtype=np.float32)
        total_gpu = gpuarray.to_gpu(total)
        kernels = SourceModule(
        """
            __global__ void summ_final(float *x, float *total){
                __shared__ float psum[1024];
                int index = threadIdx.x;
                int inext;

                psum[index] = x[index];
                __syncthreads();

                inext = blockDim.x / 2;

                while (inext >= 1){
                    if (index < inext){
                        psum[index] = psum[index] + psum[index + inext];
                    }
                    inext = inext / 2;
                    __syncthreads();
                }

                if (index == 0){
                    total[0] = psum[0];
                }
            }
        """)
        summ_final = kernels.get_function("summ_final")
        summ_final(data_gpu.gpudata, total_gpu.gpudata, block=(1024, 1, 1))
        print(total_gpu[0])


if __name__ == "__main__":
    numbers = np.ones(1024, dtype=np.float32)
    data = DataAnalysis(numbers, "numbers")
    data.plot_autocorrelation()
