from __future__ import division
from collections import OrderedDict
import logging
import sys

from nose.tools import eq_, nottest
import numpy as np
from path import path

import pycuda_helpers
print pycuda_helpers.__file__
from pycuda_helpers.cuda import MultiArrayPointers
from pycuda_helpers.shared_storage.cuda import shared_storage, shared_storage_multiarray
from . import all_close


@nottest
def _single_shared_storage_test(threads_per_block, block_count):
    '''
    Test `SharedStorage` CUDA class
    -------------------------------

    In this test, we co-operatively fill a shared array of type `int32_t`
    within a CUDA kernel by using an instance of the `SharedStorage` class.
    Each thread computes the following and stores it in the next available
    position in the data array:

        blockIdx.x * 10000 + threadIdx.x

    In addition, as part of the default behaviour of `SharedStorage`, when each
    item is appended to the `data` array, the block index and thread index are
    recorded in a separate `ThreadContext` array.  This makes it
    straight-forward to reference the data according to the thread that
    produced it.
    '''
    occupancy, data, thread_contexts = shared_storage(
            threads_per_block=threads_per_block, block_count=block_count)
    ordered_contexts = sorted(set([tuple(c) for c in thread_contexts]))

    expected_contexts = [(block_id, thread_id)
            for block_id in range(block_count)
                         for thread_id in range(threads_per_block)]
    expected_data = [(block_id * 10000 + thread_id)
            for block_id in range(block_count)
                         for thread_id in range(threads_per_block)]
    eq_(ordered_contexts, expected_contexts)
    eq_(sorted(data), expected_data)



@nottest
def _single_shared_storage_multiarray_test(threads_per_block, block_count):
    occupancy, data, thread_contexts = shared_storage_multiarray(
            threads_per_block=threads_per_block, block_count=block_count)
    import pudb; pudb.set_trace()

    ordered_contexts = sorted(set([tuple(context)
            for context in thread_contexts]))
    expected_contexts = [(block_id, thread_id)
            for block_id in range(block_count)
                         for thread_id in range(threads_per_block)]
    expected_a = np.array([(block_id, block_id)
                           for block_id in range(block_count)
                           for thread_id in range(threads_per_block)],
                          dtype=np.int32)
    expected_b = np.array([thread_id for block_id in range(block_count)
                           for thread_id in range(threads_per_block)],
                          dtype=np.uint16)

    labels = ('CPU', 'CUDA')

    # Since the order of the CUDA results is non-deterministic, sort data
    # arrays for validation.
    for data_array in [expected_a, expected_b, a, b, c]:
        data_array.sort(axis=0)

    expected_c = expected_b.astype(np.float32)

    try:
        all_close(ordered_contexts, expected_contexts, labels=labels)
        all_close(a, expected_a, labels=labels)
        all_close(b, expected_b, labels=labels)
        all_close(c, expected_c, labels=labels)
    except:
        import pudb; pudb.set_trace()


def shared_storage_test():
    for threads_per_block, block_count in [
            (32, 22),
            (1024, 1),
            (1024, 100),
            (343, 37), ]:
        yield (_single_shared_storage_multiarray_test, threads_per_block,
               block_count)
        #yield _single_shared_storage_test, threads_per_block, block_count
