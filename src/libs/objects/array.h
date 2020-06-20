#include <libs/utils/logger.h>
#include <cstring>

template <typename T>
class array
{
    size_t      a_size;
    T*          a_data;
    T*          a_end;
    bool        a_preallocated = false;

    struct iterator
    {
    public:
        T* value;

        iterator &operator++(int)
        {
            value = (T*) ((unsigned long)value + sizeof(T));
            return *this;
        }

        iterator &operator++()
        {
            value = (T*) ((unsigned long)value + sizeof(T));
            return *this;
        }

        T* operator*() noexcept
        {
            return value;
        }

        T* operator =(T* new_value)
        {
            *value = *new_value;
            return value;
        }

        T* operator =(T new_value)
        {
            *value = new_value;
            return value;
        }

        bool operator !=(T* end) const
        {
            return value != end;
        }

        bool operator !=(iterator end) const
        {
            return value != *end;
        }
    };

public:
    array(size_t size) :
        a_size(size),
        a_data((T*) calloc(size, sizeof(T))),
        a_end((T*)((long)a_data + size * sizeof(T)))
    {
        // DEBUG_LOG("[array] allocated %zu bytes memory block (%zu x %zu bytes) from #%p to #%p\n", size * sizeof(T), size, sizeof(T), a_data, a_end);
    };
    array(size_t size, T* ptr) :
        a_size(size),
        a_data(ptr),
        a_end((T*)((long)ptr + size * sizeof(T))),
        a_preallocated(true)
    {
        // DEBUG_LOG("[array] preallocated %zu bytes memory block (%zu x %zu bytes) from #%p to #%p\n", size * sizeof(T), size, sizeof(T), a_data, a_end);
    };
    array(std::initializer_list<T> list) :
        a_size(list.size()),
        a_data((T*) calloc(list.size(), sizeof(T))),
        a_end((T*)((long)a_data + list.size() * sizeof(T)))
    {
        // DEBUG_LOG("[array] allocated %zu bytes memory block (%zu x %zu bytes) from #%p to #%p with initializer_list\n", list.size() * sizeof(T), list.size(), sizeof(T), a_data, a_end);
        iterator it{a_data};
        for (auto item : list) {
            it = item;
            it++;
        }
    };
    // array(ssize_t size, T* ptr) :
    //     a_size(size),
    //     a_data(ptr),
    //     a_end((T*)((long)ptr + size * sizeof(T)))
    // {
    //     DEBUG_LOG("[array] initialized array from #%p to #%p (%zu x %zu bytes)\n", ptr, a_end, a_size, sizeof(T));
    // };
    ~array() {
        if (!a_preallocated) free(a_data);
        // DEBUG_LOG("[~array] destroy (%zu x %zu bytes)\n", a_size, sizeof(T));
    }

    size_t size() {
        return a_size;
    }

    T* data() {
        return a_data;
    }

    void set_new(T* ptr) {
        free(a_data);
        a_data = ptr;
    }

    iterator begin() const {
        return iterator{a_data};
    }

    iterator end() const {
        return iterator{a_end};
    }

};