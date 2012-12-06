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
class SharedStorageBase {
public:
    ThreadContext *thread_contexts_;
    T *data_;

    uint32_t capacity_;
    /* Note: the following must be a pointer to a device global variable. */
    uint32_t *id_;

    __device__ SharedStorageBase(uint32_t *id, uint32_t capacity,
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
};


template <class T>
class SharedStorage : public SharedStorageBase<T> {
public:
    __device__ SharedStorage(uint32_t *id, uint32_t capacity,
            ThreadContext *thread_contexts, T *data)
            : SharedStorageBase<T>(id, capacity, thread_contexts, data) {}

    __device__ T& append(T const &item) {
        uint32_t id = this->get_id();
        assert(id < this->capacity_);
        this->thread_contexts_[id] = ThreadContext(blockIdx.x, threadIdx.x);
        this->data_[id] = item;
        return this->data_[id];
    }
};

}

#endif
