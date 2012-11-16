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
    uint32_t foobar_;
};


typedef StructContainer<Foo> Bar;

extern "C" {
    __global__ void test_kernel(int32_t bar_count, Bar **bar_array) {
        if(threadIdx.x == 0 && blockIdx.x == 0) {
            for(int b = 0; b < bar_count; b++) {
                Bar &bar = *bar_array[b];
                printf("Bar.object_count: %d\n", bar.object_count());
                for(int i = 0; i < bar.object_count(); i++) {
                    bar.objects_[i].foobar_ += 10 + 100 * (b + 1);
                }
            }
        }
    }
}
''', arch='compute_20', code='sm_20', keep=True,
                 include_dirs=[get_include_root()],
                 no_extern_c=True)


class Foo(object):
    def __init__(self, foobar):
        self.foobar = foobar

    def __repr__(self):
        return '''Foo(foobar=%s)''' % self.foobar

    def __str__(self):
        return repr(self)

    def as_array(self):
        return np.array([self.foobar], dtype=np.uint32)


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
