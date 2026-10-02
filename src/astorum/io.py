import numpy as np

def writeArrayToFile(filename : str, array : np.ndarray) -> None:
    array = np.asarray(array)
    with open(filename, 'w') as file:
        if array.ndim == 1:
            file.write(f"{array.shape[0]}\n")
            row_str = ' '.join(map(str, array))
            file.write(f"{row_str}\n")
        elif array.ndim == 2:
            file.write(f"{array.shape[0]} {array.shape[1]}\n")
            for row in array:
                row_str = ' '.join(map(str, row))
                file.write(f"{row_str}\n")
        else:
            raise ValueError("Input array must be 1D or 2D")
       
        
def readArrayFromFile(filename : str) -> np.ndarray:
    with open(filename) as file:
        dims = list(map(int, file.readline().split()))
        
        if len(dims) == 1: 
            
            size = dims[0]
            array = np.zeros(size)
            
            values = list(map(float, file.readline().split()))
            array[:] = values
            
        elif len(dims) == 2:
            rows, cols = dims
            array = np.zeros((rows, cols))
            
            for i in range(rows):
                row_values = list(map(float, file.readline().split()))
                array[i, :] = row_values
                
        else:
            raise ValueError("Invalid dimension format in file.")
    
    return array