#pragma once
#include <fstream>
#include <type_traits>
#include <Eigen/Core>

namespace kevlar {

struct Serializer
{
    Serializer(const char* fname)
        : f_(fname, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc)
    {}

    template <class D>
    friend inline Serializer& operator<<(Serializer&, const D&);

private:
    std::ofstream f_;
};

template <class VecType>
inline Serializer& operator<<(
        Serializer& s, 
        const VecType& v)
{
    using value_t = std::decay_t<decltype(*v.data())>;
    s.f_.write(reinterpret_cast<const char*>(v.data()), sizeof(value_t) * v.size());
    s.f_.flush();
    return s;
}

struct UnSerializer
{
    UnSerializer(const char* fname)
        : f_(fname, std::ios_base::in | std::ios_base::binary)
    {}

    template <class D>
    friend inline UnSerializer& operator>>(UnSerializer&, D&);

private:
    std::ifstream f_;
};

template <class VecType>
inline UnSerializer& operator>>(
        UnSerializer& s,
        VecType& v
        )
{
    using value_t = std::decay_t<decltype(*v.data())>;
    s.f_.read(reinterpret_cast<char*>(v.data()), sizeof(value_t) * v.size());
    return s;
}

} // namespace kevlar
