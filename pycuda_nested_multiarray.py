import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


m = SourceModule(r'''
#include <stdint.h>
#include <stdio.h>


class Foo {
public:
    uint32_t foobar_;
};


class Bar {
public:
    Foo *foo_;
    int32_t *foo_count_;

    __device__ int32_t foo_count() const { return *foo_count_; }
};


__global__ void test_kernel(Bar *bar) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Bar.foo_count: %d\n", bar[0].foo_count());
        bar[0].foo_[0].foobar_ = 1234;
    }
}
''', arch='compute_20', code='sm_20', keep=True)


def main():
    import numpy as np

    foo_count = 3

    d_foo_array = cuda.to_device(np.array(range(foo_count), dtype=np.uint32))
    d_foo_count = cuda.to_device(np.array([foo_count], dtype=np.int32))

    bar = np.array([d_foo_array, d_foo_count], dtype=np.intp)
    bar_array = cuda.to_device(bar)

    t = m.get_function('test_kernel')
    t(bar_array, block=(1, 1, 1))


if __name__ == '__main__':
    main()
