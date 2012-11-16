from path import path
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

from struct_container import StructContainer


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
    __global__ void test_kernel(Bar *bar) {
        if(threadIdx.x == 0 && blockIdx.x == 0) {
            printf("Bar.object_count: %d\n", bar[0].object_count());
            for(int i = 0; i < bar[0].object_count(); i++) {
                bar[0].objects_[i].foobar_ += 10;
            }
        }
    }
}
''', arch='compute_20', code='sm_20', keep=True,
                 include_dirs=[path('pycuda_include').abspath()],
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


def main(foo_count):
    t = m.get_function('test_kernel')

    bar = StructContainer(Foo, [Foo(i) for i in range(foo_count)])
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
