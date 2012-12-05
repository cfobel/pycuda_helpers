#include <stdint.h>
#include "SharedStorage.hpp"

using shared_storage::SharedStorage;
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
