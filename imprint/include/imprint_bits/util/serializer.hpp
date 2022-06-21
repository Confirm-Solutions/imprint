#pragma once
#include <Eigen/Core>
#include <fstream>
#include <type_traits>

namespace imprint {

struct Serializer {
    Serializer(const char* fname)
        : f_(fname, std::ios_base::out | std::ios_base::binary |
                        std::ios_base::trunc) {}

    std::ofstream& get() { return f_; }

   private:
    std::ofstream f_;
};

struct UnSerializer {
    UnSerializer(const char* fname)
        : f_(fname, std::ios_base::in | std::ios_base::binary) {}

    std::ifstream& get() { return f_; }

   private:
    std::ifstream f_;
};

// Arithmetic type
template <class ValueType>
inline std::enable_if_t<std::is_arithmetic_v<std::decay_t<ValueType>>,
                        Serializer&>
operator<<(Serializer& s, ValueType v) {
    auto& f = s.get();
    f.write(reinterpret_cast<char*>(&v), sizeof(ValueType));
    f.flush();
    return s;
}

template <class ValueType>
inline std::enable_if_t<std::is_arithmetic_v<std::decay_t<ValueType>>,
                        UnSerializer&>
operator>>(UnSerializer& us, ValueType& v) {
    auto& f = us.get();
    f.read(reinterpret_cast<char*>(&v), sizeof(ValueType));
    return us;
}

// Eigen matrices (TODO: NOT PORTABLE RIGHT NOW DUE TO ENDIANNESS)
template <class ValueType, int R, int C, int O, int MR, int MC>
inline Serializer& operator<<(
    Serializer& s, const Eigen::Matrix<ValueType, R, C, O, MR, MC>& m) {
    using value_t = ValueType;
    uint32_t r = m.rows();
    uint32_t c = m.cols();
    auto& f = s.get();
    f.write(reinterpret_cast<char*>(&r), sizeof(uint32_t));
    f.write(reinterpret_cast<char*>(&c), sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(m.data()),
            sizeof(value_t) * m.size());
    f.flush();
    return s;
}

template <class ValueType, int R, int C, int O, int MR, int MC>
inline UnSerializer& operator>>(UnSerializer& us,
                                Eigen::Matrix<ValueType, R, C, O, MR, MC>& m) {
    using value_t = ValueType;
    uint32_t r = 0;
    uint32_t c = 0;
    auto& f = us.get();
    f.read(reinterpret_cast<char*>(&r), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&c), sizeof(uint32_t));
    m.resize(r, c);
    f.read(reinterpret_cast<char*>(m.data()), sizeof(value_t) * m.size());
    return us;
}

}  // namespace imprint
