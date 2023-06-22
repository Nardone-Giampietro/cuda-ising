import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import data_analysis as da
import linecache
from natsort import natsorted 
from alive_progress import alive_bar

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size" : "24"
})

class magn_data:
    def __init__(self, files_path, size, sus=False, mag=False, name=None):
        self._files_path = str(files_path)
        self._size = int(size)
        self._name = str(name)
        self._files_list = natsorted(os.listdir(files_path))
        self._data_size = len(self._files_list)

        self._sus_data = np.empty(self._data_size, dtype=np.float64)
        self._sus_errors = np.empty(self._data_size, dtype=np.float64)
        self._beta_data = np.empty(self._data_size, dtype=np.float64)
        self._mag_data = np.empty(self._data_size, dtype=np.float64)
        self._mag_errors = np.empty(self._data_size, dtype=np.float64)


        if sus:
            self._fill_sus_data()

        if mag:
            self._fill_mag_data()

    @property
    def size(self):
        return self._size
    
    @property
    def name(self):
        return self._name

    @property
    def beta(self):
        return self._beta_data
    
    @property
    def mag(self):
        return self._mag_data
    
    @property
    def mag_err(self):
        return self._mag_errors
    
    @property
    def sus(self):
        return self._sus_data

    @property
    def sus_err(self):
        return self._sus_errors

    def _fill_sus_data(self):
        print(f"Filling Susceptibility Data for L = {self._size} ...")
        with alive_bar(self._data_size) as bar:
            for count, file_name in enumerate(self._files_list):
                path_file = os.path.join(self._files_path, file_name)
                with open(path_file) as file:
                    data_load = np.loadtxt(file, dtype=np.float64, comments="#")
                    data = da.DataAnalysis(data_load)
                    beta_line = linecache.getline(path_file, 2)
                    first, remainder = beta_line.split(" B ")
                    beta = np.float64(remainder.split()[0])
                    self._beta_data[count] = beta
                    factor = beta * (self._size**2)
                    variance_square = data.sampleVarianceSquare
                    self._sus_data[count] = factor * variance_square
                    self._sus_errors[count] = factor * data.sampleVarianceError(
                        resamplings=50
                    )
                bar()
    
    def _fill_mag_data(self):
        print(f"Filling Magnetization Data for L = {self._size} ...")
        with alive_bar(self._data_size) as bar:
            for count, file_name in enumerate(self._files_list):
                path_file = os.path.join(self._files_path, file_name)
                with open(path_file) as file:
                    data_load = np.loadtxt(file, dtype=np.float64, comments="#")
                    data = da.DataAnalysis(data_load)
                    beta_line = linecache.getline(path_file, 2)
                    first, remainder = beta_line.split(" B ")
                    beta = np.float64(remainder.split()[0])
                    self._beta_data[count] = beta
                    self._mag_data[count] = data.mean
                    self._mag_errors[count] = data.sampleMeanVariance
                bar()


    def save_sus(self):
        file_name = self._name
        np.savetxt(
            file_name + "_SUS.txt", 
            np.c_[self._beta_data, self._sus_data, self._sus_errors], 
            delimiter=" ", 
            newline="\n", 
            header="L "+str(self._size),
            comments="# "
        )
        print(f"Susceptibility Data Saved for L = {self._size}.\n")
    
    def save_mag(self):
        file_name = self._name
        np.savetxt(
            file_name + ".txt", 
            np.c_[self._beta_data, self._mag_data, self._mag_errors], 
            delimiter=" ", 
            newline="\n", 
            header="L "+str(self._size),
            comments="# "
        )
        print(f"Magnetization Data Saved for L = {self._size}.\n")
    
    def save_aut(self, delta=0.5):
        filename = self._name
        tau_int_array = np.empty(self._data_size, dtype=np.float64)
        print(f"Computing Autocorrelation Data for L = {self._size} ...")
        with alive_bar(self._data_size) as bar:
            for count, file_name in enumerate(self._files_list):
                path_file = os.path.join(self._files_path, file_name)
                with open(path_file) as file:
                    data_load = np.loadtxt(file, dtype=np.float64, comments="#")
                    data = da.DataAnalysis(data_load)
                    beta_line = linecache.getline(path_file, 2)
                    first, remainder = beta_line.split(" B ")
                    beta = np.float64(remainder.split()[0])
                    self._beta_data[count] = beta
                    tau_int_array[count] = data.intAutocorrelationTime(delta=delta)
                bar()
        
        np.savetxt(
            filename + "_AUT.txt", 
            np.c_[self._beta_data, tau_int_array], 
            delimiter=" ", 
            newline="\n", 
            header="L "+str(self._size),
            comments="# "
        )
        print(f"Autocorrelation Data Saved for L = {self._size}.\n")


def plot_mg(*args):
    color_map = ["b", "g", "r", "c", "m", "y", "k", "w"]
    w, h = 2 * figaspect(1/2)
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\left< \left| M \right| \right>$")
    ax.autoscale(enable=True, axis="both")
    for count, data in enumerate(args):
        size_line = linecache.getline(data, 1)
        first, remainder = size_line.split(" L ")
        size = remainder.split()[0]
        x, y, yerr = np.loadtxt(data, dtype=np.float64, comments="#", unpack=True)
        ax.errorbar(x, y, yerr, 
                    fmt=color_map[count] + "o", 
                    elinewidth=1, 
                    capsize=5,
                    markersize= 4,
                    label= r"L = "+size+"",
                    )
    plt.grid(visible=True, axis="both", linestyle="--")
    plt.axvline(x=0.4406868,
              color = "b",
              linestyle="--",
              label = r"$\beta_{c}$"
              )
    plt.title(r"Magnetizzazione", pad=20.0)
    ax.legend()
    plt.show()

def plot_sus(*args):
    color_map = ["b", "g", "r", "c", "m", "y", "k", "w"]
    w, h = 2 * figaspect(1/2)
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\chi_{M}$")
    ax.autoscale(enable=True, axis="both")
    for count, data in enumerate(args):
        size_line = linecache.getline(data, 1)
        first, remainder = size_line.split(" L ")
        size = remainder.split()[0]
        x, y, yerr = np.loadtxt(data, dtype=np.float64, comments="#", unpack=True)
        ax.errorbar(x, y, yerr, 
                    fmt=color_map[count] + "o", 
                    elinewidth=1, 
                    capsize=5,
                    markersize= 4,
                    label= r"L = "+size+"",
                    )
    plt.grid(visible=True, axis="both", linestyle="--")
    plt.title(r"Suscettività Magnetica $\chi_M$", pad=20.0)
    ax.legend()
    plt.show()

def plot_aut(*args):
    color_map = ["b", "g", "r", "c", "m", "y", "k", "w"]
    w, h = 2 * figaspect(1/2)
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\tau_{int}$")
    ax.autoscale(enable=True, axis="both")
    for count, data in enumerate(args):
        size_line = linecache.getline(data, 1)
        first, remainder = size_line.split(" L ")
        size = remainder.split()[0]
        x, y= np.loadtxt(data, dtype=np.float64, comments="#", unpack=True)
        plt.plot(x, y, 
                    color_map[count] + "-",
                    markersize= 4,
                    label= r"L = "+size+"",
                    )
    plt.grid(visible=True, axis="both", linestyle="--")
    plt.title(r"Lunghezze di Autocorrelazione", pad=20.0)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    current_dir = os.getcwd()
    
    dir8 = current_dir + "/MG_L8/"
    dir20 = current_dir + "/MG_L20/"
    dir30 = current_dir + "/MG_L30/"
    dir40 = current_dir + "/MG_L40/"

    # mg8 = magn_data(dir8, 8, name="MG_L8", mag=True)
    # mg8.save_mag()
    #mg8.save_aut(delta=1.0)

    # mg20 = magn_data(dir20, 20, name="MG_L20", mag=True)
    # mg20.save_mag()
    #mg20.save_aut(delta=1.0)

    # mg30 = magn_data(dir30, 30, name="MG_L30", mag=True)
    # mg30.save_mag()
    #mg30.save_aut(delta=1.0)

    mg40 = magn_data(dir40, 40, name="MG_L40", mag=True)
    mg40.save_mag()
    #mg40.save_aut(delta=0.5)

    plot_mg(current_dir + "/MG_L8.txt",
             current_dir + "/MG_L20.txt",
             current_dir + "/MG_L30.txt",
             current_dir + "/MG_L40.txt"
            )
