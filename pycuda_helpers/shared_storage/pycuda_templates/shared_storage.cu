#include <stdint.h>
#include <assert.h>
#include "SharedStorage.hpp"

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


class TestData {
public:
    int32_t *a_;
    uint16_t *b_;
    float *c_;
};


class SharedTestMultiArray : public SharedStorageBase<TestData> {
public:
    __device__ SharedTestMultiArray(uint32_t *id, uint32_t capacity,
            ThreadContext *thread_contexts, TestData *data)
            : SharedStorageBase<TestData>(id, capacity, thread_contexts, data) {
    }

    __device__ void append(int32_t a, uint16_t b, float c) {
        uint32_t id = this->get_id();
        assert(id < this->capacity_);
        this->thread_contexts_[id] = ThreadContext(blockIdx.x, threadIdx.x);
        this->data_->a_[id] = a;
        this->data_->b_[id] = b;
        this->data_->c_[id] = c;
    }
};


extern "C" __global__ void test_shared_storage_multiarray(uint32_t *capacity_ptr,
        ThreadContext *thread_contexts, TestData *data) {
    /*SharedTestMultiArray<TestData> storage(&g__shared_storage_id, *capacity_ptr,*/
            /*thread_contexts, data);*/
    //storage.append(blockIdx.x * 10000 + threadIdx.x);
    SharedTestMultiArray storage(&g__shared_storage_id, *capacity_ptr,
            thread_contexts, data);
    storage.append(blockIdx.x, threadIdx.x, threadIdx.x);
}
