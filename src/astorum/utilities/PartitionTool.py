import os
import numpy as np
import hyperspy.api as hs
import exspy
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.widgets import Button
from dask.distributed import Client, LocalCluster, get_client
from scipy.linalg import svd
from typing import List, Tuple, Optional
import psutil
import warnings

def ordinal(value : int) -> str:
    """
    Converts zero or a *postive* integer (or their string
    representations) to an ordinal value.

    >>> for i in range(1,13):
    ...     ordinal(i)
    ...
    '1st'
    '2nd'
    '3rd'
    '4th'
    '5th'
    '6th'
    '7th'
    '8th'
    '9th'
    '10th'
    '11th'
    '12th'

    >>> for i in (100, '111', '112',1011):
    ...     ordinal(i)
    ...
    '100th'
    '111th'
    '112th'
    '1011th'

    Notes
    -----
    Author:  Serdar Tumgoren
    https://code.activestate.com/recipes/576888-format-a-number-as-an-ordinal/
    MIT license
    """
    try:
        value = int(value)
    except ValueError:
        return value

    if value % 100 // 10 != 1:
        if value % 10 == 1:
            ordval = "%d%s" % (value, "st")
        elif value % 10 == 2:
            ordval = "%d%s" % (value, "nd")
        elif value % 10 == 3:
            ordval = "%d%s" % (value, "rd")
        else:
            ordval = "%d%s" % (value, "th")
    else:
        ordval = "%d%s" % (value, "th")

    return ordval


def initDask(workers : Optional[int] = None) -> Client:
    if workers is None:
        workers = os.cpu_count()
    
    try:
        client = get_client()
    except ValueError:
        cluster = LocalCluster(n_workers = workers, threads_per_worker = 1)
        client = Client(cluster)
    
    print("Dask client initialised successfully.")
    print("Number of workers:", workers)
    print("IP address:", client.dashboard_link)
    
    return client

def findOptimalBlockSize(array : np.ndarray, blockSizeFactor : Optional[int] = 1, tol : Optional[float] = 0.2) -> Tuple[int, int, int]:
    dim0, dim1, dim2 = array.shape
    itemSize = array.dtype.itemsize
    
    avaliableMemory = int(psutil.virtual_memory().available * 0.8 // os.cpu_count()) # 80% of available memory per worker in MB
    targetBlockSize = int(avaliableMemory / blockSizeFactor)

    blockSize = array.blocks[0, 0, 0].nbytes
    originalAspectRatio = dim0 / dim1
    limDim0 = int(np.floor(np.sqrt(dim0)))
    limDim1 = int(np.floor(np.sqrt(dim1)))
    
    deltaAspectRatio = np.inf
    
    optimalBlock = array.chunksize

    for i in range(dim0, limDim0, -1):
        for j in range(dim1, limDim1, -1):
            localBlock = (i, j, dim2)
            blockSize = np.prod(localBlock) * itemSize
            localAspectRatio = i / j
            localDeltaAspectRatio = abs(localAspectRatio - originalAspectRatio)
        
            if ((1 - tol) * targetBlockSize <= blockSize <= (1 + tol) * targetBlockSize):
                if (localDeltaAspectRatio < deltaAspectRatio):
                    deltaAspectRatio = localDeltaAspectRatio
                    optimalBlock = localBlock
                    
        return optimalBlock
    
def normalisePoissonianNoise(array : np.ndarray) -> np.ndarray:
    if (array < 0).any():
        raise ValueError("Data contains negative values, cannot normalise Poissonian noise.")
    
    G = array.sum(axis = 1)
    H = array.sum(axis = 0)
    sqrtG = np.sqrt(G)[:, np.newaxis]
    sqrtH = np.sqrt(H)[np.newaxis, :]
    
    with (np.errstate(divide = 'ignore', invalid = 'ignore')):
        norm_array = array / (sqrtG * sqrtH)
        norm_array = np.nan_to_num(norm_array)
    
    return norm_array


class PartitionedEDXSDataset:
    def __init__(self, dataset : exspy.signals.LazyEDSTEMSpectrum, components : Optional[int] = 30, blocksize : Optional[Tuple[int, int]] = None) -> None:
        self.dataset = dataset
        self.data = dataset.data
        self.data = self.data.rechunk((blocksize[0], blocksize[1], self.data.shape[2]))
        self.components = components
        self.blockstructure = self.data.blocks.shape
        self.blockshape = self.data.blocks[0, 0, 0].shape
        self.explained_variances = np.zeros(shape = (self.blockstructure[0], self.blockstructure[1], self.components))
        self.explained_variance_ratios = np.zeros(shape = (self.blockstructure[0], self.blockstructure[1], self.components))
        self.relevant_components = np.zeros(shape = (self.blockstructure[0], self.blockstructure[1]), dtype = np.int32)
        self.max_components = 0
        
        
    def partition(self, blocksize : Tuple[int, int]) -> None:
        self.data = self.data.rechunk(blocksize)
        self.blockstructure = self.data.blocks.shape
        self.blockshape = self.data.blocks[0, 0, 0].shape
        self.explained_variances = np.zeros(shape = (self.blockstructure[0], self.blockstructure[1], self.components))
        self.explained_variance_ratios = np.zeros(shape = (self.blockstructure[0], self.blockstructure[1], self.components))
        self.relevant_components = np.zeros(shape = (self.blockstructure[0], self.blockstructure[1]), dtype = np.int32)
        
        
    def plotBlocks(self, **kwargs) -> None:
        fig, axs = plt.subplots(nrows = self.blockstructure[0], ncols = self.blockstructure[1], figsize = (12, 12))

        for i in range(self.blockstructure[0]):
            for j in range(self.blockstructure[1]):
                block_data = self.data.blocks[i, j, 0].sum(axis = 2).compute()

                im = axs[i, j].imshow(block_data, interpolation = "nearest", **kwargs)

                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].set_xticklabels([])
                axs[i, j].set_yticklabels([])

                for spine in axs[i, j].spines.values():
                    spine.set_edgecolor("white")
                    spine.set_linewidth(2)

                axs[i, j].set_title(f"Block ({i}, {j})")

        plt.subplots_adjust(wspace = 0.1, hspace = 0.1)

        cbar = fig.colorbar(im, ax = axs, orientation = "vertical", pad = 0.05, fraction = 0.02)
        cbar.set_label("Intensity")
        fig.suptitle("Block Structure (summed over energy axis)")

        plt.show()
        
        
    def estimateElbowPosition(self, D_block : np.ndarray, explained_variance_ratio : np.ndarray, log : Optional[bool] = True, max_points : Optional[int] = 20) -> int:
        max_points = min(max_points, len(explained_variance_ratio) - 1)
        adj = np.clip(explained_variance_ratio, 1e-30, None)
        
        x1 = 0
        x2 = max_points
        
        if log:
            y1 = np.log(adj[0])
            y2 = np.log(adj[max_points])
        else:
            y1 = adj[0]
            y2 = adj[max_points]
        
        kw = {}
        kw['like'] = D_block
        xs = np.arange(max_points, **kw)
        
        if log:
            ys = np.log(adj[:max_points])
        else:
            ys = adj[:max_points]
        
        numer = abs((x2 - x1) * (y1 - ys) - (x1 - xs) * (y2 - y1))
        denom = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distance = np.nan_to_num(numer / denom)
        elbow_position = np.argmax(distance)
        
        return elbow_position
        
        
    def blockWiseSVD(self, normalise_poissonian_noise : Optional[bool] = False, **kwargs) -> None:
        for i in range(self.blockstructure[0]):
            for j in range(self.blockstructure[1]):
                D_flat = self.data.blocks[i, j, 0].reshape(self.blockshape[0] * self.blockshape[1], self.blockshape[2]).compute()
                
                if normalise_poissonian_noise:
                    D_flat = normalisePoissonianNoise(D_flat)
                    
                U, S, V = svd(a = D_flat, **kwargs)
                U = U[:, :self.components]
                S = S[:self.components]
                V = V[:self.components, :]
                self.explained_variances[i, j, :] = S ** 2 / D_flat.shape[0]
                self.explained_variance_ratios[i, j, :] = self.explained_variances[i, j, :] / self.explained_variances[i, j, :].sum()
                self.relevant_components[i, j] = self.estimateElbowPosition(D_flat, self.explained_variance_ratios[i, j, :])
        
        self.max_components = np.max(self.relevant_components)
        
        
    def inspectSpectrum(self, block_index : Tuple[int, int], **kwargs) -> None:
        temp = hs.signals.EDSTEMSpectrum(self.data.blocks[block_index[0], block_index[1], 0].compute())
        temp.metadata = self.dataset.metadata
        temp.axes_manager = self.dataset.axes_manager
        temp.axes_manager[0].size = self.blockshape[0]
        temp.axes_manager[1].size = self.blockshape[1]
        temp.plot()
    
    
    def inspectSummedSpectrum(self, block_index : Tuple[int, int], **kwargs) -> None:
        temp = hs.signals.EDSTEMSpectrum(self.data.blocks[block_index[0], block_index[1], 0].compute())
        temp.metadata = self.dataset.metadata
        temp.axes_manager = self.dataset.axes_manager
        temp.axes_manager[0].size = self.blockshape[0]
        temp.axes_manager[1].size = self.blockshape[1]
        temp.sum().plot()

           
    def plotExplainedVarianceRatios(self, n : Optional[int] = 30, log : Optional[bool] = True, xaxis_type : Optional[str] = "index", xaxis_labeling : Optional[str] = None, signal_fmt : Optional[dict] = None, noise_fmt : Optional[dict] = None, **kwargs) -> None:
        fig, axs = plt.subplots(nrows = self.explained_variance_ratios.shape[0], ncols = self.explained_variance_ratios.shape[1], figsize = (18, 18))

        for i in range(self.explained_variance_ratios.shape[0]):
            for j in range(self.explained_variance_ratios.shape[1]):
                block_explained_variance_ratio = self.explained_variance_ratios[i, j, :]
                
                n_max = len(block_explained_variance_ratio)
                
                if self.components is None:
                    n = n_max
                elif n > n_max:
                    warnings.warn("n is too large, setting n to its maximal value.")
                    n = n_max
                    
                n_signal_pcs = self.relevant_components[i, j]
                cutoff = block_explained_variance_ratio[n_signal_pcs - 1]
                
                if signal_fmt is None:
                    signal_fmt = {
                        "c": "#C24D52",
                        "linestyle": "",
                        "marker": "^",
                        "markersize": 5,
                        "zorder": 3,
                    }

                if noise_fmt is None:
                    noise_fmt = {
                        "c": "#4A70B0",
                        "linestyle": "",
                        "marker": "o",
                        "markersize": 5,
                        "zorder": 3,
                    }

                if xaxis_labeling is None:
                    xaxis_labeling = "cardinal" if xaxis_type == "index" else "ordinal"

                axes_titles = {
                    "y": "Proportion of variance",
                    "x": f"Principal component {xaxis_type}",
                }
                
                if n < n_max:
                    block_explained_variance_ratio = block_explained_variance_ratio[:n]
                    
                if log:
                    axs[i, j].set_yscale("log")

                axs[i, j].axvline(self.relevant_components[i, j], linewidth = 2, color = "gray", linestyle = "dashed", zorder = 1)
                
                index_offset = 0
                
                if xaxis_type == "number":
                    index_offset = 1
                
                if n_signal_pcs == n:
                    axs[i, j].plot(range(index_offset, index_offset + n), block_explained_variance_ratio[:n], **signal_fmt)
                
                elif n_signal_pcs > 0:
                    axs[i, j].plot(range(index_offset, index_offset + n_signal_pcs), block_explained_variance_ratio[:n_signal_pcs], **signal_fmt)
                    axs[i, j].plot(range(index_offset + n_signal_pcs, index_offset + n), block_explained_variance_ratio[n_signal_pcs:n], **noise_fmt)

                else:
                    axs[i, j].plot(range(index_offset, index_offset + n), block_explained_variance_ratio[:n], **noise_fmt)

                if xaxis_labeling == "ordinal":
                    axs[i, j].xaxis.set_major_formatter(FuncFormatter(lambda x, p: ordinal(x)))

                axs[i, j].set_title(f"Block ({i}, {j})")
                axs[i, j].set_xlabel(axes_titles["x"])
                axs[i, j].set_ylabel(axes_titles["y"])
                axs[i, j].xaxis.set_major_locator(MaxNLocator(integer = True, min_n_ticks = 1))
                # axs[i, j].text(0.5, -0.1, f"Relevant components : {self.relevant_components[i, j].compute()}", transform = axs[i, j].transAxes,
                #        ha = 'center', va = 'center', fontsize = 6, color = 'black')
                axs[i, j].margins(0.05)
                axs[i, j].autoscale()
                
        fig.suptitle("Explained Variance Ratios")
        
        plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
        plt.show()