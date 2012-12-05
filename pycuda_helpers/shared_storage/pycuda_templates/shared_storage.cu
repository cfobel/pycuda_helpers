#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "SharedStorage.hpp"
#include "Point.hpp"
#include "SwapConfig.hpp"

using shared_storage::SharedStorage;
using shared_storage::SharedStorageBase;
using shared_storage::ThreadContext;

{% if not c_types -%}
{%- set c_types=["int32_t"] -%}
{%- endif -%}

__device__ uint32_t g__shared_storage_id = 0;

{% for c_type in c_types %}
{% set i=loop.counter0 %}
{% set count=loop.counter0 %}
#define COUNT {{ count }}
typedef {{ c_type }} C_{{i}};
extern "C" __global__ void test_shared_storage_{{ c_type }}(uint32_t *capacity_ptr,
        ThreadContext *thread_contexts, C_{{i}} *data) {
    SharedStorage<C_{{i}}> storage(&g__shared_storage_id, *capacity_ptr,
            thread_contexts, data);
    storage.append(blockIdx.x * 10000 + threadIdx.x);
}
{% endfor %}

class NetDeltaCostInfo {
public:
    MoveData<Point<float> > sum_;
    MoveData<Point<float> > sq_sum_;
    uint32_t cardinality_;
    int32_t net_id_;
};

class TestData {
public:
    MoveData<int32_t> *ids_;
    MoveData<Point<int32_t> > *coords_;

    /* Use `uint8_t` instead of `bool` to avoid special handling in case `bool`
     * instances are allocated differently on CUDA/CPU. */
    uint8_t *master_;
    uint8_t *participate_;
    uint8_t *accepted_;
};


class SharedTestMultiArray : public SharedStorageBase<TestData> {
public:
    __device__ SharedTestMultiArray(uint32_t *id, uint32_t capacity,
            ThreadContext *thread_contexts, TestData *data)
            : SharedStorageBase<TestData>(id, capacity, thread_contexts, data) {
    }

    __device__ void append(MoveData<int32_t> const &ids,
            MoveData<Point<int32_t> > const &coords, uint8_t const &master,
            uint8_t const &participate, uint8_t const &accepted) {
        uint32_t id = this->get_id();
        assert(id < this->capacity_);
        this->thread_contexts_[id] = ThreadContext(blockIdx.x, threadIdx.x);
        this->data_->ids_[id] = ids;
        this->data_->coords_[id] = coords;
        this->data_->master_[id] = master;
        this->data_->participate_[id] = participate;
        this->data_->accepted_[id] = accepted;
    }
};


extern "C" __global__ void test_shared_storage_multiarray(uint32_t *capacity_ptr,
        ThreadContext *thread_contexts, TestData *data) {
    /*SharedTestMultiArray<TestData> storage(&g__shared_storage_id, *capacity_ptr,*/
            /*thread_contexts, data);*/
    //storage.append(blockIdx.x * 10000 + threadIdx.x);
    SharedTestMultiArray storage(&g__shared_storage_id, *capacity_ptr,
            thread_contexts, data);
    storage.append(
            MoveData<int32_t>(blockIdx.x, blockIdx.x), 
            MoveData<Point<int32_t> >(Point<int32_t>(blockIdx.x, blockIdx.x),
                    Point<int32_t>(threadIdx.x, threadIdx.x)),
            blockIdx.x % 2,
            threadIdx.x % 2,
            threadIdx.x % 4);
}
