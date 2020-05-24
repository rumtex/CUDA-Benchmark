#include <libs/utils/logger.h>
#include <cstring>

struct S {
    explicit (S)(const S&);    // error in C++20, OK in C++17
    explicit (operator int)(); // error in C++20, OK in C++17
};

template <typename T>
class array
{
    ssize_t  a_size;
    T*      a_data;
    T*      a_end;

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
    };

public:
    array(ssize_t size) :
        a_size(size),
        a_data((T*) valloc(size * sizeof(T))),
        a_end((T*)((long)a_data + size * sizeof(T)))
    {
        // DEBUG_LOG("[array] allocated %zu bytes memory block (%zu x %zu bytes) from #%p to #%p\n", size * sizeof(T), size, sizeof(T), a_data, a_end);
    };
    array(std::initializer_list<T> list) :
        a_size(list.size()),
        a_data((T*) valloc(list.size() * sizeof(T))),
        a_end((T*)((long)a_data + list.size() * sizeof(T)))
    {
        // DEBUG_LOG("[array] allocated %zu bytes memory block (%zu x %zu bytes) from #%p to #%p with initializer_list\n", list.size() * sizeof(T), list.size(), sizeof(T), a_data, a_end);
        iterator it{a_data};
        for (auto item : list) {
            it = item;
            it++;
        }
    };
    array(ssize_t size, T* ptr) :
        a_size(size),
        a_data(ptr),
        a_end((T*)((long)ptr + size * sizeof(T)))
    {
        // DEBUG_LOG("[array] initialized array from #%p to #%p (%zu x %zu bytes)\n", ptr, a_end, a_size, sizeof(T));
    };

    ~array() = default;

    ssize_t size() {
        return a_size;
    }

    T* data() {
        return a_data;
    }

    iterator begin() const {
        return iterator{a_data};
    }

    T* end() const {
        return a_end;
    }

};