import sys
from pprint import pprint

from path import path
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

sys.path.insert(0, path(__file__).abspath().parent)
from pycuda_helpers.struct_container import StructContainer
from pycuda_helpers import get_include_root


m = SourceModule(r'''
#include <stdint.h>
#include <stdio.h>
#include "StructContainer.hpp"

class Foo {
public:
    uint32_t *foobar_;
    int32_t *data_;
};


typedef StructContainer<Foo> Bar;

extern "C" {
    __global__ void test_kernel(int32_t bar_count, Bar **bar_array) {
        if(threadIdx.x == 0 && blockIdx.x == 0) {
            for(int b = 0; b < bar_count; b++) {
                Bar &bar = *bar_array[b];
                printf("Bar.object_count: %d\n", bar.object_count());
                for(int i = 0; i < bar.object_count(); i++) {
                    uint32_t &foobar = *(bar.objects_[i].foobar_);
                    foobar += 10 + 100 * (b + 1);
                }
            }
        }
    }
}
''', arch='compute_20', code='sm_20', keep=True,
                 include_dirs=[get_include_root()],
                 no_extern_c=True)


class Foo(object):
    def __init__(self, foobar, data=None):
        self.foobar = foobar
        if data is None:
            data = range(10)
        self.data = np.array(data, dtype=np.int32)
        self.d_foobar = None
        self.d_data = None

    def __repr__(self):
        return '''Foo(foobar=%s, data=%s)''' % (self.foobar, self.data)

    def __str__(self):
        return repr(self)

    @classmethod
    def from_array(cls, a):
        foobar_array = cuda.from_device(a[0], 1, dtype=np.uint32)
        data = cuda.from_device(a[1], 10, dtype=np.int32)
        return cls(foobar_array[0], data)

    def as_array(self):
        self.d_foobar = cuda.to_device(np.array([self.foobar], dtype=np.uint32))
        self.d_data = cuda.to_device(self.data)
        return np.array([self.d_foobar, self.d_data], dtype=np.intp)


def main(foo_count, bar_count=5):
    t = m.get_function('test_kernel')

    bar_list = [StructContainer(Foo, [Foo(i) for i in range(foo_count)])
                for b in range(bar_count)]
    bar_arrays = [bar.sync_to_device() for bar in bar_list]

    pprint(bar_list)

    t(np.int32(len(bar_list)), cuda.In(np.array(bar_arrays, dtype=np.intp)),
        block=(len(bar_list), 1, 1))

    for bar in bar_list:
        bar.sync_from_device()
    pprint(bar_list)


def test():
    main(10)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        raise SystemExit, 'usage: %s foo_count' % sys.argv[0]
    main(int(sys.argv[1]))
