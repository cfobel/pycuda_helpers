#ifndef ___SHARED_STORAGE__HPP___
#define ___SHARED_STORAGE__HPP___

//#define DEBUG_SHARED_STORAGE

namespace shared_storage {

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>


class ThreadContext {
/*
 * This class enables a shared array of type `T` to be filled co-operatively
 * within a CUDA kernel by serializing write access to subsequent positions.
 *
 * In addition, as part of the default behaviour of `SharedStorage`, when each
 * item is appended to the `data_` array, the block index and thread index are
 * recorded in a separate `ThreadContext` array.  This makes it
 * straight-forward to reference the data according to the thread that produced
 * it.
 */
public:
    uint32_t block_idx_;
    uint32_t thread_idx_;

    __device__ ThreadContext() : block_idx_(0), thread_idx_(0) {}

    __device__ ThreadContext(uint32_t block_idx, uint32_t thread_idx) :
            block_idx_(block_idx), thread_idx_(thread_idx) {}
};


template <class T>
class SharedStorage {
public:
    ThreadContext *thread_contexts_;
    T *data_;

    uint32_t capacity_;
    /* Note: the following must be a pointer to a device global variable. */
    uint32_t *id_;

    __device__ SharedStorage(uint32_t *id, uint32_t capacity,
            ThreadContext *thread_contexts, T *data)
            : id_(id), capacity_(capacity), thread_contexts_(thread_contexts),
            data_(data) {}

    __device__ uint32_t get_id() {
        /* Equivalent to:
         *
         *     value = *id_;
         *     *id_ += 1;
         */
        uint32_t value = atomicAdd(id_, 1);
        return value;
    }

    __device__ T& append(T const &item) {
        uint32_t id = get_id();
        assert(id < capacity_);
        thread_contexts_[id] = ThreadContext(blockIdx.x, threadIdx.x);
        data_[id] = item;
        return data_[id];
    }
};

}

#endif
