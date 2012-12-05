from __future__ import division

from nose.tools import eq_, nottest
from ..shared_storage.cuda import shared_storage


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


def single_shared_storage_test():
    for threads_per_block, block_count in [
            (1024, 1),
            (32, 22),
            (1024, 100),
            (343, 37), ]:
        yield _single_shared_storage_test, threads_per_block, block_count
