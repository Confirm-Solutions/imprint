#pragma once
#include <type_traits>
#include <kevlar_bits/util/types.hpp>

namespace kevlar {

template <class ValueType=double
        , class UIntType=uint32_t>
struct GridRange;

template <class ValueType=double
        , class UIntType=uint32_t>
struct GridptViewer
{
    using value_t = ValueType;
    using uint_t = UIntType;

    GridptViewer(
        size_t dim,
        value_t* ptheta,
        value_t* pradius,
        uint_t* psim_size)
        : theta_(ptheta, dim)
        , radius_(pradius, dim)
        , sim_size_(psim_size)
    {}

    auto& get_theta() { return theta_; }
    auto& get_radius() { return radius_; }
    auto& get_sim_size() { return *sim_size_; }

    void reset(
            value_t* ptheta, 
            value_t* pradius,
            uint_t* psim_size)
    {
        new (&theta_) Eigen::Map<vec_t>(
                ptheta, theta_.size());
        new (&radius_) Eigen::Map<vec_t>(
                pradius, radius_.size());
        sim_size_ = psim_size;
    }

private: 
    using vec_t = std::conditional_t<
            std::is_const_v<value_t>,
            const colvec_type<std::decay_t<value_t> >,
            colvec_type<value_t>
        >;
    Eigen::Map<vec_t> theta_;
    Eigen::Map<vec_t> radius_;
    uint_t* sim_size_;
};

template <class ValueType
        , class UIntType>
struct GridRange
{
    using value_t = ValueType;
    using uint_t = UIntType;

    struct iterator_type 
    {
        using difference_type = std::ptrdiff_t;
        using value_type = GridptViewer<value_t, uint_t>;
        using pointer = GridptViewer<value_t, uint_t>*;
        using reference = GridptViewer<value_t, uint_t>&;
        using iterator_category = std::random_access_iterator_tag;

        iterator_type(GridRange& outer,
                      size_t cnt)
            : outer_ref_{outer}
            , viewer_(outer.dim(),
                      outer.thetas_.data() + cnt*outer.dim(),
                      outer.radii_.data() + cnt*outer.dim(),
                      outer.sim_sizes_.data()+cnt)
            , cnt_{cnt}
        {}

        iterator_type& operator+=(difference_type n) {
            cnt_ += n;
            auto& outer = outer_ref_.get();
            viewer_.reset(
                    viewer_.get_theta().data() + n*outer.dim(),
                    viewer_.get_radius().data() + n*outer.dim(),
                    outer.sim_sizes_.data() + cnt_);
            return *this;
        }
        iterator_type& operator++() { 
            ++cnt_; 
            auto& outer = outer_ref_.get();
            viewer_.reset(
                    viewer_.get_theta().data() + outer.dim(),
                    viewer_.get_radius().data() + outer.dim(),
                    outer.sim_sizes_.data() + cnt_);
            return *this; 
        }
        reference operator*() { return viewer_; }
        pointer operator->() { return &viewer_; }

        difference_type operator-(const iterator_type& it2)
        {
            return cnt_ - it2.cnt_;
        }

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
        GridptViewer<value_t, uint_t> viewer_;
        size_t cnt_;
    };

    struct const_iterator_type 
    {
        using difference_type = std::ptrdiff_t;
        using value_type = GridptViewer<const value_t, const uint_t>;
        using pointer = const GridptViewer<const value_t, const uint_t>*;
        using reference = const GridptViewer<const value_t, const uint_t>&;
        using iterator_category = std::random_access_iterator_tag;

        const_iterator_type(
                const GridRange& outer,
                size_t cnt)
            : outer_ref_{outer}
            , viewer_(outer.dim(),
                      outer.thetas_.data() + cnt*outer.dim(),
                      outer.radii_.data() + cnt*outer.dim(),
                      outer.sim_sizes_.data()+cnt)
            , cnt_{cnt}
        {}

        const_iterator_type& operator+=(difference_type n) {
            cnt_ += n;
            auto& outer = outer_ref_.get();
            viewer_.reset(
                    viewer_.get_theta().data() + n*outer.dim(),
                    viewer_.get_radius().data() + n*outer.dim(),
                    outer.sim_sizes_.data() + cnt_);
            return *this;
        }
        const_iterator_type& operator++() { 
            ++cnt_; 
            const auto& outer = outer_ref_.get();
            viewer_.reset(
                    viewer_.get_theta().data() + outer.dim(),
                    viewer_.get_radius().data() + outer.dim(),
                    outer.sim_sizes_.data() + cnt_);
            return *this; 
        }
        reference operator*() { return viewer_; }
        pointer operator->() { return &viewer_; }

        inline constexpr bool 
        operator==(const const_iterator_type& it2) const
        { 
            return (this->cnt_ == it2.cnt_) &&
                (&this->outer_ref_.get() == &it2.outer_ref_.get()); 
        }

        inline constexpr bool 
        operator!=(const const_iterator_type& it2) const
        { 
            return (this->cnt_ != it2.cnt_) ||
                (&this->outer_ref_.get() != &it2.outer_ref_.get()); 
        }

    private:
        std::reference_wrapper<const GridRange> outer_ref_;
        GridptViewer<const value_t, const uint_t> viewer_;
        size_t cnt_;
    };

    GridRange() =default;

    GridRange(
        uint_t dim,
        uint_t size)
        : thetas_(dim, size)
        , radii_(dim, size)
        , sim_sizes_(size)
    {
        sim_sizes_.setZero();
    }

    mat_type<value_t>& get_thetas() { return thetas_; }
    const mat_type<value_t>& get_thetas() const { return thetas_; }
    mat_type<value_t>& get_radii() { return radii_; }
    const mat_type<value_t>& get_radii() const { return radii_; }
    colvec_type<uint_t>& get_sim_sizes() { return sim_sizes_; }
    const colvec_type<uint_t>& get_sim_sizes() const { return sim_sizes_; }

    iterator_type begin() { return {*this, 0}; }
    iterator_type end() { return {*this, size()}; }
    const_iterator_type begin() const { return {*this, 0}; }
    const_iterator_type end() const { return {*this, size()}; }
    size_t size() const { return thetas_.cols(); }
    size_t dim() const { return thetas_.rows(); }

private:
    mat_type<value_t> thetas_;          // matrix of theta vectors
    mat_type<value_t> radii_;           // matrix of radius vectors
    colvec_type<uint_t> sim_sizes_;      // vector of simulation sizes
};

} // namespace kevlar
