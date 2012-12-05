from __future__ import division

import numpy as np

from cuda_common import get_cuda_function, cuda


def shared_storage(threads_per_block=1024, block_count=1):
    dtype = np.dtype('int32')
    test = get_cuda_function('shared_storage.cu', 'test_shared_storage', dtype)
    total_thread_count = threads_per_block * block_count
    data = np.empty(total_thread_count, dtype=dtype)
    thread_contexts = np.empty((total_thread_count, 2), dtype=np.uint32)

    # Store capacity as array so kernel can pass back final occupancy count.
    capacity = np.array([total_thread_count], dtype=np.uint32)

    block = (threads_per_block, 1, 1)
    grid = (block_count, 1, 1)

    print 'thread_count: %d' % threads_per_block
    print 'block_count: %d' % block_count

    test(cuda.InOut(capacity), cuda.Out(thread_contexts), cuda.Out(data),
            block=block, grid=grid)
    return capacity[0], data, thread_contexts


if __name__ == '__main__':
    shared_storage()
