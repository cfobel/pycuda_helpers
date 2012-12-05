import numpy as np
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    PYCUDA_ENABLED = True
except ImportError:
    # PyCUDA is not available
    PYCUDA_ENABLED = False


class MultiArrayPointers(object):
    @classmethod
    def mem_size(self, items):
        try:
            iter(items)
            items = len(items)
        finally:
            return items * np.intp(0).nbytes

    def __init__(self, arrays, struct_ptr=None, copy_to=True):
        # If struct_ptr is not set, we need to allocate the memory manually
        if struct_ptr is None:
            self.struct_ptr = cuda.mem_alloc(self.mem_size(len(arrays)))
            self._cleanup_struct = True
        else:
            self.struct_ptr = struct_ptr
            self._cleanup_struct = False
        self.data = []
        for a in arrays:
            if copy_to:
                alloc_func = cuda.to_device
            else:
                alloc_func = cuda.mem_alloc
            try:
                self.data.append(alloc_func(a.size * a.itemsize))
            except TypeError:
                self.data.append(alloc_func(np.array([a], dtype=a.dtype)))
        self.shapes = [a.shape for a in arrays]
        self.dtypes = [a.dtype for a in arrays]
        start_ptr = int(self.struct_ptr)
        for i, a in enumerate(arrays):
            cuda.memcpy_htod(start_ptr, np.intp(int(self.data[i])))
            start_ptr += np.intp(0).nbytes

    def get_from_device(self, index_list=None):
        '''
        Copy array data from GPU device and wrap in a numpy arrays.

        If index_list is None, return list of numpy arrays (one/array).
        If index_list is a single integer, return single numpy array.
        If index_list is an iterable, list of numpy arrays
                (one/selected array).
        '''
        single = False
        if index_list is None:
            index_list = range(len(self.data))
        else:
            try:
                int(index_list)
                index_list = [index_list]
                single = True
            except TypeError:
                pass
        results = []
        try:
            for i in index_list:
                results.append(cuda.from_device(self.data[i], self.shapes[i],
                            self.dtypes[i]))
        except cuda.LaunchError:
            import traceback
            traceback.print_exc()
            traceback.print_stack()
            raise ValueError, 'Invalid device pointer: %d' % i
        if single:
            return results[0]
        else:
            return results

    def __str__(self):
        return str([cuda.from_device(self.data[i], self.shapes[i],
                self.dtypes[i]) for i in range(len(self.data))])


def get_integer_type_map(min_bits=8, max_bits=32, unsigned=True, signed=True):
    # Create a mapping from un/signed 8/16/32 bit integer numpy types to stdint
    # types.
    signed_variants = []
    if unsigned:
        signed_variants += ['u']
    if signed:
        signed_variants += ['']
    numpy_type_names = set(['%sint%d' % (signed, bits)
            for signed in signed_variants
                    for bits in [8, 16, 32]
                            if bits >= min_bits and bits <= max_bits])
    return dict([(np_dtype, '%s_t' % np_dtype)
            for np_dtype in numpy_type_names])


# Pre-compute commonly used integer type maps
all_integer_types_map = get_integer_type_map()
signed_integer_types_map = get_integer_type_map(unsigned=False)
unsigned_integer_types_map = get_integer_type_map(signed=False)


def get_float_type_map():
    # Create a mapping from 32/64 bit float numpy types to CUDA types
    # types.
    return {'float32': 'float', 'float64': 'double'}

all_float_types_map = get_float_type_map()


def validate_dtype(allowed_types, dtype):
    try:
        dtype.name
    except AttributeError:
        dtype = dtype(0).dtype
    finally:
        if dtype.name not in allowed_types:
            raise TypeError, 'Type %s not supported.  Supported types: %s' % (
                dtype, ', '.join(['numpy.%s' % k for k in allowed_types]))
    return dtype
