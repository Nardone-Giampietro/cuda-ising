import logging
import sys
import numpy as np
from pycuda.compiler import SourceModule
from pycuda import autoinit
from pycuda import gpuarray
from pycuda.reduction import ReductionKernel
import matplotlib.pyplot as plt


class DataAnalysis:
    kernels_cu = open("kernels.cu", "r")
    kernels_cu = kernels_cu.read()

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
        self._dim = len(data)
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
        times = np.arange(self._dim, dtype=np.int32)
        plt.plot(
            times, self._data, "ro", scalex=scalex, scaley=scaley, data=data, **kwargs
        )
        plt.show()

    def __summ_reduction(self, array, dim, offset=0, product=False):
        BlockDim = 256
        GridDim = 1024
        BlockGridDim = int(self._dim / (BlockDim * GridDim))

        kernels = SourceModule(self.kernels_cu)
        summ_final = kernels.get_function("summ_final")
        summ_partial = kernels.get_function("summ_partial")
        summ_product_partial = kernels.get_function("summ_product_partial")

        dim = np.int32(dim)
        offset = np.int32(offset)

        data_gpu = gpuarray.to_gpu(array)
        sum_total = gpuarray.zeros(1, dtype=np.float32)
        partial = gpuarray.empty(GridDim, dtype=np.float32)

        if not product:
            summ_partial(
                data_gpu.gpudata,
                partial.gpudata,
                dim,
                offset,
                block=(BlockDim, 1, 1),
                grid=(GridDim, BlockGridDim),
            )
        else:
            summ_product_partial(
                data_gpu.gpudata,
                partial.gpudata,
                dim,
                offset,
                block=(BlockDim, 1, 1),
                grid=(GridDim, BlockGridDim),
            )

        summ_final(partial.gpudata, sum_total.gpudata, block=(GridDim, 1, 1))

        return sum_total[0]

    def plot_autocorrelation(self, max_distance=3000, scalex=True, scaley=True, data=None, **kwargs):
        autocorrelations = gpuarray.zeros(max_distance + 1, dtype=np.float32)

        for offset in np.arange(max_distance + 1):
            sample_dimension = np.float32(self._dim - offset)
            product_sum = self.__summ_reduction(self._data, self._dim, offset=offset, product=True).get()
            simple_sum = self.__summ_reduction(self._data, self._dim, product=False).get()
            offset_sum = self.__summ_reduction(self._data, self._dim, offset=offset, product=False).get()
            autocorrelations[offset] = (product_sum / sample_dimension) - simple_sum * offset_sum / (sample_dimension**2)
        
        times = np.arange(max_distance + 1, dtype=np.int32)
        plt.plot(times, autocorrelations.get() / autocorrelations.get()[0], "-r")
        plt.show()

if __name__ == "__main__":
    Data= np.loadtxt("Magn_L10_B03_01.txt", dtype=np.float32)
    data = DataAnalysis(Data, "numbers")
    data.plot_autocorrelation()
