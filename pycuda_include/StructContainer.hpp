template <class T>
class StructContainer {
public:
    T *objects_;
    int32_t *object_count_;

    __device__ int32_t object_count() const { return *object_count_; }
};
