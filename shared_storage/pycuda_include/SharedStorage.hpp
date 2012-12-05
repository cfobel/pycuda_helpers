#ifndef ___SHARED_STORAGE__HPP___
#define ___SHARED_STORAGE__HPP___

//#define DEBUG_SHARED_STORAGE

namespace shared_storage {

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

template <class T>
class SharedStorage {
public:
    T *data_;

    uint32_t capacity_;
    /* Note: the following must be a pointer to a device global variable. */
    uint32_t *id_;

    __device__ SharedStorage(uint32_t *id, uint32_t capacity, T *data)
            : id_(id), capacity_(capacity), data_(data) {}

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
        data_[id] = item;
        return data_[id];
    }
};

}

#endif
