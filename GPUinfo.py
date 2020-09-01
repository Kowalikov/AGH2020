import time
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import math


# -- initialize the device
import pycuda.autoinit

dev = pycuda.autoinit.device
print(dev.get_attribute(pycuda.driver.device_attribute.WARP_SIZE))
print(dev.get_attribute(pycuda.driver.device_attribute.MAX_BLOCK_DIM_X))
print(dev.get_attribute(pycuda.driver.device_attribute.MAX_BLOCK_DIM_Y))
print(dev.get_attribute(pycuda.driver.device_attribute.MAX_BLOCK_DIM_Z))
print(dev.get_attribute(pycuda.driver.device_attribute.MAX_GRID_DIM_X))
print(dev.get_attribute(pycuda.driver.device_attribute.MAX_GRID_DIM_Y))
print(dev.get_attribute(pycuda.driver.device_attribute.MAX_GRID_DIM_Z))
print(dev.get_attribute(pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK))