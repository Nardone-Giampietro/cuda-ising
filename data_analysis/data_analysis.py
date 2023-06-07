import logging
import sys
import numpy as np
from pycuda.compiler import SourceModule
from pycuda import autoinit
from pycuda import gpuarray
from pycuda import curandom
from pycuda.reduction import ReductionKernel
import matplotlib.pyplot as plt


class DataAnalysis:
    kernels_cu = open("kernels.cu", "r")
    kernels_cu = kernels_cu.read()

    def __init__(self, data, name=None, max_distance=0):
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

        self._data = np.array(data, dtype=np.float64)
        self._dim = len(data)
        self._name = str(name)

        self._rng = curandom.XORWOWRandomNumberGenerator(curandom.seed_getter_uniform)

        if max_distance != 0:
            self._max_distance = max_distance
            self._autocorrelations = self.__autocorrelations(max_distance)

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._name
    
    @property
    def mean(self):
        return self.__simpleMean(self._data)
    
    @property
    def squareMean(self):
        return self.__squareMean(self._data)
    
    @property
    def unbiasedNaiveMeanVariance(self):
        return np.sqrt(self.__unbiasedNaiveMeanVariance(self._data))
    
    @property
    def sampleVariance(self):
        return np.sqrt(self.__sampleVariance(self._data))

    @property  
    def sampleMeanVariance(self):
        sampleMeanVariance = (self.__unbiasedNaiveMeanVariance(self._data)) * (
            1.0 + 2.0 * self.intAutocorrelationTime()
        )
        return np.sqrt(sampleMeanVariance)
    
    def sampleVarianceError(self, resamplings=1):
        variances = self.__varianceBootstrap(resamplings=resamplings)
        variance_error = np.sqrt(self.__sampleVariance(variances))
        return variance_error

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return self._dim

    def __getitem__(self, key):
        return self._data[key]

    def __str__(self):
        return str(self._data)

    def __summ_reduction(self, array, dim, offset=0, product=False):
        BlockDim = 512
        GridDim = 1024

        if (dim <= BlockDim * GridDim):
            summ = np.float64(0)
            for i in np.arange(dim - offset):
                if product:
                    summ = summ + array[i] * array[i + offset]
                else:
                    summ = summ + array[i + offset]
            return summ
        else:
            BlockGridDim = int(dim / (BlockDim * GridDim))
            kernels = SourceModule(self.kernels_cu)
            summ_final = kernels.get_function("summ_final")
            summ_partial = kernels.get_function("summ_partial")
            summ_product_partial = kernels.get_function("summ_product_partial")

            dim = np.int32(dim)
            offset = np.int32(offset)

            data_gpu = gpuarray.to_gpu(array)
            sum_total = gpuarray.zeros(1, dtype=np.float64)
            partial = gpuarray.empty(GridDim, dtype=np.float64)

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

            return sum_total.get()[0]

    def __simpleMean(self, array):
        simpleMean = self.__summ_reduction(
            array, len(array), offset=0, product=False
        )
        simpleMean = simpleMean / np.float64(len(array))
        return simpleMean


    def __squareMean(self, array):
        squareMean = self.__summ_reduction(
            array, len(array), offset=0, product=True
        )
        squareMean = squareMean / np.float64(len(array))
        return squareMean

    def __sampleVariance(self, array):
        sampleVariance = self.__squareMean(array) - self.__simpleMean(array)**2
        return sampleVariance

    def __unbiasedNaiveMeanVariance(self, array):
        unbiasedNaiveMeanVariance = self.__sampleVariance(array) / np.float64(len(array) - 1)
        return unbiasedNaiveMeanVariance
    
    def __resample(self, array):
        dim = len(array)
        BlockDim = 1024
        BlockGridDim = int(dim / BlockDim)

        kernels = SourceModule(self.kernels_cu)
        resample = kernels.get_function("resample")
        array_gpu = gpuarray.to_gpu(array)
        array_out_gpu = gpuarray.empty(dim, dtype=np.float64)
        numbers_gpu = self._rng.gen_uniform(dim, dtype=np.float32) * (dim - 1)
        numbers_gpu = numbers_gpu.astype(np.int32) 

        resample(
            array_gpu.gpudata,
            np.int32(dim),
            numbers_gpu.gpudata,
            array_out_gpu.gpudata,
            block=(BlockDim, 1, 1),
            grid=(BlockGridDim, 1, 1)
        )

        return array_out_gpu.get()
    
    def __dataBlocking(self, array, steps=1):
        try:
            if (2**steps >= len(array)):
                raise ValueError
        except ValueError:
            logging.exception(
                "Data Blocking iterations exceeded the dimension of the data"
            )
            sys.exit()

        BlockDim = 1024

        kernels = SourceModule(self.kernels_cu)
        array_reduction = kernels.get_function("array_reduction")

        dataBlockingVariances = np.empty(steps, dtype=np.float64)

        array_gpu = gpuarray.to_gpu(array)
        dim = len(array)
        next_array_gpu = gpuarray.empty(int(dim / 2), dtype=np.float64)

        for step in np.arange(steps):
            if (len(array_gpu) <= BlockDim):
                BlockGridDim = 1
            else:
                BlockGridDim = int(len(array_gpu) / BlockDim)
            
            array_reduction(
                array_gpu.gpudata,
                np.int32(dim),
                next_array_gpu.gpudata,
                block=(BlockDim, 1, 1),
                grid=(BlockGridDim, 1, 1)
            )
            dim = int(dim / 2)
            array = next_array_gpu.get()
            array_gpu = next_array_gpu
            next_array_gpu = gpuarray.empty(int(dim / 2), dtype=np.float64)
            dataBlockingVariances[step] = np.sqrt(self.__unbiasedNaiveMeanVariance(array))

        return dataBlockingVariances

    def __autocorrelations(self, max_distance):
        autocorrelations_gpu = gpuarray.zeros(max_distance + 1, dtype=np.float64)

        for offset in np.arange(max_distance + 1):
            sample_dimension = np.float64(self._dim - offset)
            product_sum = self.__summ_reduction(
                self._data, self._dim, offset=offset, product=True
            )
            simple_sum = self.__summ_reduction(
                self._data, self._dim, product=False
            )
            offset_sum = self.__summ_reduction(
                self._data, self._dim, offset=offset, product=False
            )
            autocorrelations_gpu[offset] = (
                product_sum / sample_dimension
            ) - simple_sum * offset_sum / (sample_dimension**2)

        autocorrelations = autocorrelations_gpu.get()

        return autocorrelations
    
    def __varianceBootstrap(self, resamplings=1):
        variances = np.empty(resamplings, dtype=np.float64)

        for step in np.arange(resamplings):
            resampled_array = self.__resample(self._data)
            variances[step] = self.__sampleVariance(resampled_array)
        
        return variances
    
    def plotVarianceBootstrap(self, max_resamplings=1):
        variances = np.empty(max_resamplings, dtype=np.float64)
        
        for resamplings in np.arange(1, max_resamplings + 1):
            variances_array = self.__varianceBootstrap(resamplings=resamplings)
            variances[resamplings - 1] = np.sqrt(self.__sampleVariance(variances_array))
        
        plt.xlabel("Number of resamplings")
        plt.ylabel("Estimate of Sus Error ")
        resamplings = np.arange(1, max_resamplings + 1, 1, dtype=np.int32)
        plt.plot(resamplings, variances, "r+")
        plt.show()


    def intAutocorrelationTime(self):
        tau_int = np.float64(0.0)
        for auto in self._autocorrelations[1::]:
            tau_int = tau_int + auto
        tau_int = tau_int / self._autocorrelations[0]
        return int(tau_int)

    def plot(self, scalex=True, scaley=True, data=None, **kwargs):
        plt.xlabel("Time t")
        plt.ylabel(self._name)
        times = np.arange(self._dim, dtype=np.int32)
        plt.plot(
            times, self._data, "ro", scalex=scalex, scaley=scaley, data=data, **kwargs
        )
        plt.show()

    def plotAutocorrelation(self, scalex=True, scaley=True, data=None, **kwargs):
        autocorr = self._autocorrelations

        plt.xlabel("Time t")
        plt.ylabel(f"{self._name} autocorrelation")
        times = np.arange(self._max_distance + 1, dtype=np.int32)
        plt.plot(times, autocorr / autocorr[0], "-r")
        plt.show()

    def plotDataBlocking(self, steps=1, scalex=True, scaley=True, data=None, **kwargs):
        dataBlockingVariances = self.__dataBlocking(self._data, steps=steps)

        plt.xlabel("Iterations")
        plt.ylabel("Variances")
        iterations = np.arange(1, steps + 1, 1, dtype=np.int32)
        plt.xticks(iterations)
        plt.semilogy()
        plt.plot(iterations, dataBlockingVariances, "r+")
        plt.show()

