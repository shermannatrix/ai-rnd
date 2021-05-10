from __future__ import division
from numba import cuda
import numpy
import math

@cuda.jit
def my_kernel(io_array):
	
	# Compute flatten index inside the array
	pos = cuda.grid(1)
	if pos < io_array.size:	# Check array boundaries
		io_array[pos] *= 2	# do the computation

# Create the data array - usually initialized some other way
data = numpy.ones(256)

# Set the number of threads in a block
threadsperblock = 256

# Calculate the number of thread blocks in the grid
blockspergrid = math.ceil(data.shape[0] / threadsperblock)

# Now start the kernel
my_kernel[blockspergrid, threadsperblock](data)

# Print the result
print(data)