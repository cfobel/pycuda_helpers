import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np


class StructContainer(object):
    '''
    This class acts as a helper to pass an array of structs or classes to a
    PyCuda kernel.

    The C++ class code to implement this class in a CUDA kernel is in the
    header `pycuda_include/StructContainer.hpp`.

    :param object_class: The class/struct type of the objects to be stored.
    :param object_list: An optional list of objects to initialize the container.
    '''
    def __init__(self, object_class, object_list=None):
        self.object_class = object_class
        if object_list is None:
            self.object_list = []
        else:
            self.object_list = object_list
        self.object_array = None
        self.d_object_array = None
        self.device_ptr = None

    def __repr__(self):
        return '''StructContainer(object_list=%s)''' % self.object_list

    def __str__(self):
        return repr(self)

    @property
    def object_count(self):
        return len(self.object_list)

    def append(self, obj):
        self.object_list.append(obj)

    def sync_to_device(self):
        self.object_array = np.array([f.as_array()
                for f in self.object_list])
        self.d_object_array = cuda.to_device(self.object_array)
        self.d_object_count = cuda.to_device(np.array([self.object_count],
                                                   dtype=np.int32))
        self.device_ptr = cuda.to_device(np.array([self.d_object_array,
                                                   self.d_object_count],
                                                  dtype=np.intp))
        return self.device_ptr

    def sync_from_device(self):
        object_array = cuda.from_device_like(self.d_object_array,
                                             self.object_array)
        self.object_list = [self.object_class(*obj) for obj in object_array]
