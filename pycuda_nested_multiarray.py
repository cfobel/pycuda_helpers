import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np


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
        for(int i = 0; i < bar[0].foo_count(); i++) {
            bar[0].foo_[i].foobar_ += 10;
        }
    }
}
''', arch='compute_20', code='sm_20', keep=True)


class Foo(object):
    def __init__(self, foobar):
        self.foobar = foobar

    def __repr__(self):
        return '''Foo(foobar=%s)''' % self.foobar

    def __str__(self):
        return repr(self)

    def as_array(self):
        return np.array([self.foobar], dtype=np.uint32)


class Bar(object):
    def __init__(self, foo_list=None):
        if foo_list is None:
            self.foo_list = []
        else:
            self.foo_list = foo_list
        self.foo_array = None
        self.d_foo_array = None
        self.device_ptr = None

    def __repr__(self):
        return '''Bar(foo_list=%s)''' % self.foo_list

    def __str__(self):
        return repr(self)

    @property
    def foo_count(self):
        return len(self.foo_list)

    def add_foo(self, foo):
        self.foo_list.append(foo)

    def sync_to_device(self):
        self.foo_array = np.array([f.as_array()
                for f in self.foo_list])
        self.d_foo_array = cuda.to_device(self.foo_array)
        self.d_foo_count = cuda.to_device(np.array([self.foo_count],
                                                   dtype=np.int32))
        self.device_ptr = cuda.to_device(np.array([self.d_foo_array, self.d_foo_count], dtype=np.intp))
        return self.device_ptr

    def sync_from_device(self):
        foo_array = cuda.from_device_like(self.d_foo_array, self.foo_array)
        self.foo_list = [Foo(*f) for f in foo_array]


def main(foo_count):
    t = m.get_function('test_kernel')

    bar = Bar([Foo(i) for i in range(foo_count)])
    bar_array = bar.sync_to_device()

    print bar

    t(bar_array, block=(1, 1, 1))

    bar.sync_from_device()
    print bar


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        raise SystemExit, 'usage: %s foo_count' % sys.argv[0]
    main(int(sys.argv[1]))
