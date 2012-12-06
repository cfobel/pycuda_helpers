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


def np_sort_rows(a, inplace=True):
    if len(a.shape) == 1:
        result = np.sort(a)
    elif len(a.shape) > 1:
        result = np.sort(a.view([('', a.dtype)] * a.shape[1]), 0).view(a.dtype)
    else:
        raise ValueError, 'Array must have one or more dimension'

    if inplace:
        a[:] = result[:]
    return result


@nottest
def _single_shared_storage_multiarray_test(threads_per_block, block_count):
    occupancy, data, thread_contexts = shared_storage_multiarray(
            threads_per_block=threads_per_block, block_count=block_count)

    labels = ('CPU', 'CUDA')

    contexts = np.array(list(set([tuple(context)
            for context in thread_contexts])))
    expected_contexts = np.array([(block_id, thread_id)
            for block_id in range(block_count)
                         for thread_id in range(threads_per_block)])
    np_sort_rows(contexts)
    np_sort_rows(expected_contexts)
    np.allclose(contexts, expected_contexts)

    expected = OrderedDict()
    expected['ids'] = np.array([(block_id, block_id)
                           for block_id in range(block_count)
                           for thread_id in range(threads_per_block)],
                          dtype=np.int32)
    expected['coords'] = np.array([(block_id, block_id, thread_id, thread_id, )
                           for block_id in range(block_count)
                           for thread_id in range(threads_per_block)],
                          dtype=np.uint32)
    expected['master'] = np.array([block_id % 2
                           for block_id in range(block_count)
                           for thread_id in range(threads_per_block)],
                          dtype=np.uint8)
    expected['accepted'] = np.array([thread_id % 2
                           for block_id in range(block_count)
                           for thread_id in range(threads_per_block)],
                          dtype=np.uint8)
    expected['participate'] = np.array([thread_id % 4
                           for block_id in range(block_count)
                           for thread_id in range(threads_per_block)],
                          dtype=np.uint8)

    for key in data.keys():
        # Since the order of the CUDA results is non-deterministic, sort data
        # arrays for validation.
        np_sort_rows(data[key])
        np_sort_rows(expected[key])
        eq_(data[key].size, expected[key].size)
        eq_(expected[key].shape, data[key].shape)
        all_close(expected[key], data[key],
                  labels=tuple(['%s (%s)' % (key, l) for l in labels]))


def shared_storage_test():
    for threads_per_block, block_count in [
            (32, 22),
            (1024, 1),
            (1024, 100),
            (343, 37), ]:
        yield (_single_shared_storage_multiarray_test, threads_per_block,
               block_count)
        #yield _single_shared_storage_test, threads_per_block, block_count
