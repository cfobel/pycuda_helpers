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


template <class T, uint32_t ELEMENT_COUNT>
struct WrappedArray {
    T data_[ELEMENT_COUNT];

    __device__ WrappedArray() {}
    __device__ WrappedArray(T const &value) {
        clear(value);
    }

    __device__ uint32_t length() const {
        return sizeof(data_) / sizeof(data_[0]);
    }

    __device__ void clear(T const &value) {
        for(int i = 0; i < length(); i++) {
            data_[i] = value;
        }
    }
};


typedef MoveData<Point<float> > NetCost;
typedef WrappedArray<int32_t, 10> NetIdArray;
typedef WrappedArray<NetCost, 10> NetCostArray;
typedef WrappedArray<uint32_t, 10> CardinalityArray;


class TestData {
public:
    MoveData<int32_t> *ids_;
    MoveData<Point<uint32_t> > *coords_;

    /* Use `uint8_t` instead of `bool` to avoid special handling in case `bool`
     * instances are allocated differently on CUDA/CPU. */
    uint8_t *master_;
    uint8_t *participate_;
    uint8_t *accepted_;
    NetCostArray *sum_arrays_;
    NetCostArray *sq_sum_arrays_;
    CardinalityArray *cardinality_arrays_;
    NetIdArray *net_id_arrays_;
};


class SharedTestMultiArray : public SharedStorageBase<TestData> {
public:
    __device__ SharedTestMultiArray(uint32_t *id, uint32_t capacity,
            ThreadContext *thread_contexts, TestData *data)
            : SharedStorageBase<TestData>(id, capacity, thread_contexts, data) {
    }

    __device__ void append(MoveData<int32_t> const &ids,
            MoveData<Point<uint32_t> > const &coords, uint8_t master,
            uint8_t participate, uint8_t accepted,
            NetCostArray const &sum_array,
            NetCostArray const &sq_sum_array,
            CardinalityArray const &cardinality_array,
            NetIdArray const &net_id_array) {
        uint32_t id = this->get_id();
        assert(id < this->capacity_);
        this->thread_contexts_[id] = ThreadContext(blockIdx.x, threadIdx.x);
        this->data_->ids_[id] = ids;
        this->data_->coords_[id] = coords;
        this->data_->master_[id] = master;
        this->data_->participate_[id] = participate;
        this->data_->accepted_[id] = accepted;

        this->data_->sum_arrays_[id] = sum_array;
        this->data_->sq_sum_arrays_[id] = sq_sum_array;
        this->data_->cardinality_arrays_[id] = cardinality_array;
        this->data_->net_id_arrays_[id] = net_id_array;
    }
};


extern "C" __global__ void test_shared_storage_multiarray(uint32_t *capacity_ptr,
        ThreadContext *thread_contexts, TestData *data) {
    SharedTestMultiArray storage(&g__shared_storage_id, *capacity_ptr,
            thread_contexts, data);
    storage.append(
            MoveData<int32_t>(blockIdx.x, blockIdx.x), 
            MoveData<Point<uint32_t> >(Point<uint32_t>(blockIdx.x, blockIdx.x),
                    Point<uint32_t>(threadIdx.x, threadIdx.x)),
            blockIdx.x % 2,
            threadIdx.x % 2,
            threadIdx.x % 4,
            NetCostArray(NetCost(Point<float>(blockIdx.x, threadIdx.x),
                                 Point<float>(threadIdx.x, blockIdx.x))),
            NetCostArray(NetCost(Point<float>(100 * blockIdx.x, 100 *
                                              threadIdx.x),
                                 Point<float>(100 * threadIdx.x, 100 *
                                              blockIdx.x))),
            CardinalityArray(blockIdx.x), NetIdArray(threadIdx.x));
}
