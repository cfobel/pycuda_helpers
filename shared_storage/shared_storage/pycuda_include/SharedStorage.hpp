#ifndef ___SHARED_STORAGE__HPP___
#define ___SHARED_STORAGE__HPP___

//#define DEBUG_SHARED_STORAGE

namespace shared_storage {

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>


class ThreadContext {
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
