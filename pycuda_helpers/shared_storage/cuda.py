from __future__ import division
from collections import OrderedDict

import numpy as np

from pycuda_helpers.cuda import MultiArrayPointers
from cuda_common import get_cuda_function, cuda


def shared_storage(threads_per_block=1024, block_count=1):
    dtype = np.dtype('int32')
    test = get_cuda_function('shared_storage.cu', 'test_shared_storage', dtype, keep=True)
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


def shared_storage_multiarray(threads_per_block, block_count):
    test = get_cuda_function('shared_storage.cu',
                             'test_shared_storage_multiarray')

    total_thread_count = threads_per_block * block_count
    data = OrderedDict([
        ('ids', np.empty((total_thread_count, 2), dtype=np.int32)),
        ('coords', np.empty((total_thread_count, 4), dtype=np.uint32)),
        ('master', np.empty(total_thread_count, dtype=np.uint8)),
        ('accepted', np.empty(total_thread_count, dtype=np.uint8)),
        ('participate', np.empty(total_thread_count, dtype=np.uint8)),

        ('sum_arrays', np.empty((total_thread_count, 10, 4),
                                     dtype=np.float32)),
        ('sq_sum_arrays', np.empty((total_thread_count, 10, 4),
                                     dtype=np.float32)),
        ('cardinality_arrays', np.empty((total_thread_count, 10),
                                         dtype=np.uint32)),
        ('net_id_arrays', np.empty((total_thread_count, 10), dtype=np.int32)),
    ])
    data_struct = MultiArrayPointers(data.values(), copy_to=False)
    thread_contexts = np.ones((total_thread_count, 2), dtype=np.uint32)

    # Store capacity as array so kernel can pass back final occupancy count.
    capacity = np.array([total_thread_count], dtype=np.uint32)

    block = (threads_per_block, 1, 1)
    grid = (block_count, 1, 1)

    print 'thread_count: %d' % threads_per_block
    print 'block_count: %d' % block_count

    test(cuda.InOut(capacity), cuda.InOut(thread_contexts),
         data_struct.struct_ptr, block=block, grid=grid)
    for key, value in zip(data.keys(), data_struct.get_from_device()):
        data[key] = value
    return capacity[0], data, thread_contexts


if __name__ == '__main__':
    shared_storage()
