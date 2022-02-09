#pragma once
#include <type_traits>
#include <kevlar_bits/util/types.hpp>

namespace kevlar {

template <class ValueType=double
        , class IntType=uint32_t>
struct GridRange;

template <class ValueType=double
        , class IntType=uint32_t>
struct GridptViewer
{
    using value_t = ValueType;
    using int_t = IntType;

    GridptViewer(
        size_t dim,
        value_t* ptheta,
        value_t* pradius,
        int_t* sim_size,
        int_t* sim_size_rem)
        : theta_(ptheta, dim)
        , radius_(pradius, dim)
        , sim_size_(sim_size)
        , sim_size_rem_(sim_size_rem)
    {}

    auto& get_theta() { return theta_; }
    auto& get_radius() { return radius_; }
    auto& get_sim_size() { return *sim_size_; }
    auto& get_sim_size_rem() { return *sim_size_rem_; }

private: 
    Eigen::Map<colvec_type<value_t> > theta_;
    Eigen::Map<colvec_type<value_t> > radius_;
    int_t* sim_size_;
    int_t* sim_size_rem_;
};

template <class ValueType
        , class IntType>
struct GridRange
{
    using value_t = ValueType;
    using int_t = IntType;

    struct iterator_type 
    {
        using difference_type = void;
        using value_type = GridptViewer<value_t, int_t>;
        using pointer = GridptViewer<value_t, int_t>*;
        using reference = GridptViewer<value_t, int_t>&;
        using iterator_category = std::forward_iterator_tag;

        iterator_type(GridRange& outer,
                      size_t cnt)
            : outer_ref_{outer}
            , viewer_(outer.dim(),
                      outer.thetas_.data() + cnt*outer.dim(),
                      outer.radii_.data() + cnt*outer.dim(),
                      outer.sim_sizes_.data()+cnt,
                      outer.sim_sizes_rem_.data()+cnt)
            , cnt_{cnt}
        {}

        iterator_type& operator++() { 
            ++cnt_; 
            auto& outer = outer_ref_.get();
            new (&viewer_) GridptViewer<value_t, int_t>(
                    outer.dim(),
                    outer.thetas_.data() + cnt_*outer.dim(),
                    outer.radii_.data() + cnt_*outer.dim(),
                    outer.sim_sizes_.data()+cnt_,
                    outer.sim_sizes_rem_.data()+cnt_
                    );
            return *this; 
        }
        reference operator*() { return viewer_; }
        pointer operator->() { return &viewer_; }

        inline constexpr bool 
        operator==(const iterator_type& it2) const
        { 
            return (this->cnt_ == it2.cnt_) &&
                (&this->outer_ref_.get() == &it2.outer_ref_.get()); 
        }

        inline constexpr bool 
        operator!=(const iterator_type& it2) const
        { 
            return (this->cnt_ != it2.cnt_) ||
                (&this->outer_ref_.get() != &it2.outer_ref_.get()); 
        }

    private:
        std::reference_wrapper<GridRange> outer_ref_;
        GridptViewer<value_t, int_t> viewer_;
        size_t cnt_;
    };

    GridRange() =default;

    GridRange(
        int_t dim,
        int_t size)
        : thetas_(dim, size)
        , radii_(dim, size)
        , sim_sizes_(size)
        , sim_sizes_rem_(size)
    {
        sim_sizes_.setZero();
        sim_sizes_rem_.setZero();
    }

    mat_type<value_t>& get_thetas() { return thetas_; }
    const mat_type<value_t>& get_thetas() const { return thetas_; }
    auto& get_radii() { return radii_; }
    const auto& get_radii() const { return radii_; }
    auto& get_sim_sizes() { return sim_sizes_; }
    const auto& get_sim_sizes() const { return sim_sizes_; }
    auto& get_sim_sizes_rem() { return sim_sizes_rem_; }
    const auto& get_sim_sizes_rem() const { return sim_sizes_rem_; }

    iterator_type begin() { return {*this, 0}; }
    iterator_type end() { return {*this, size()}; }
    size_t size() const { return thetas_.cols(); }
    size_t dim() const { return thetas_.rows(); }

private:
    mat_type<value_t> thetas_;          // matrix of theta vectors
    mat_type<value_t> radii_;           // matrix of radius vectors
    colvec_type<int_t> sim_sizes_;      // vector of simulation sizes
    colvec_type<int_t> sim_sizes_rem_;  // vector of simulation sizes remaining
};

} // namespace kevlar
