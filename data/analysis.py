import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

class DataAnalysis:

    def __init__(self, data, name):

        try:
            iterator = iter(data)
        except TypeError:
            logging.exception(f"Argument 'data' of {type(self).__name__} is not iterable.")
            sys.exit()

        try:
            if (len(data) == 0):
                raise ValueError
        except ValueError:
            logging.exception(f"Imput data of {type(self).__name__} is empty.")
            sys.exit()
        
        self._data = np.array(data, dtype=np.float64)
        self._len = len(data)
        self._name = str(name)
    
    @property
    def data(self):
        return self._data
    
    @property
    def name(self):
        return self._name
    
    def plot_simple(self, scalex=True, scaley=True, data=None, **kwargs):
        plt.xlabel("Time t")
        plt.ylabel(self._name)
        times = np.arange(self._len, dtype=np.int32)
        plt.plot(times, self._data, "ro", scalex=scalex, scaley=scaley, data=data, **kwargs)
        plt.show()
    

if __name__ == "__main__":
    energy_data = np.loadtxt("Energy_L10_B03_01.txt", dtype=np.float64)
    Energy = DataAnalysis(energy_data, "Energy")
    Energy.plot_simple(markersize=0.1)
