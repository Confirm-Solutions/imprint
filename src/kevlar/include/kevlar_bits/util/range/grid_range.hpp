#pragma once
#include <type_traits>
#include <kevlar_bits/util/types.hpp>
#include <kevlar_bits/util/d_ary_int.hpp>

namespace kevlar {

struct rectangular_range;

struct rectangular_range
{
    struct iterator_type 
    {
        using difference_type = void;
        using value_type = dAryInt;
        using pointer = dAryInt*;
        using reference = dAryInt&;
        using iterator_category = std::forward_iterator_tag;

        iterator_type(const rectangular_range& outer,
                      size_t cnt)
            : outer_cref_{outer}, idxer_{outer.idxer_}, cnt_{cnt}
        {}

        iterator_type& operator++() { ++idxer_; ++cnt_; return *this; }
        reference operator*() { return idxer_; }
        pointer operator->() { return &idxer_; }
        
        friend constexpr bool operator==(
                const iterator_type&,
                const iterator_type&);
        friend constexpr bool operator!=(
                const iterator_type&,
                const iterator_type&);
    private:
        std::reference_wrapper<const rectangular_range> outer_cref_;
        dAryInt idxer_;
        size_t cnt_;
    };

    rectangular_range(size_t base,
                      size_t n_bits,
                      size_t size)
        : idxer_(base, n_bits), size_{size}
    {}
    
    rectangular_range(const dAryInt& idxer,
                      size_t size)
        : idxer_{idxer}, size_{size}
    {}

    iterator_type begin() const { return {*this, 0}; }
    iterator_type end() const { return {*this, size_}; }
    size_t size() const { return size_; }

    void set_idxer(const dAryInt& idx) { idxer_ = idx; }
    void set_size(size_t s) { size_ = s; }
    dAryInt& get_idxer() { return idxer_; }

private:
    dAryInt idxer_;
    size_t size_ = 0;
};

inline constexpr bool 
    operator==(const typename rectangular_range::iterator_type& it1,
               const typename rectangular_range::iterator_type& it2)
    { return (it1.cnt_ == it2.cnt_) &&
             (&it1.outer_cref_.get() == &it2.outer_cref_.get()); }

inline constexpr bool 
    operator!=(const typename rectangular_range::iterator_type& it1,
               const typename rectangular_range::iterator_type& it2)
    { return (it1.cnt_ != it2.cnt_) ||
             (&it1.outer_cref_.get() != &it2.outer_cref_.get()); }


} // namespace kevlar
