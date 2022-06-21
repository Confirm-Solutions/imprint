#pragma once
#include <algorithm>

namespace imprint {

/*
 * Sort columns of a matrix m.
 * Assumed that m is column-wise matrix.
 */
template <class MatType>
inline void sort_cols(MatType&& m) {
    auto r = m.rows();
    for (int i = 0; i < m.cols(); ++i) {
        auto m_i = m.col(i);
        std::sort(m_i.data(), m_i.data() + r);
    }
}

/*
 * Sort columns of a matrix m.
 * Assumed that m is column-wise matrix.
 * Uses a custom comparator.
 */
template <class MatType, class Comp>
inline void sort_cols(MatType&& m, Comp comp) {
    auto r = m.rows();
    for (int i = 0; i < m.cols(); ++i) {
        auto m_i = m.col(i);
        std::sort(m_i.data(), m_i.data() + r, comp);
    }
}

/*
 * Stores counts of x < elements of p into counts.
 *
 * @param   x       matrix with each column sorted.
 * @param   p       vector of sorted thresholds to check against each column of
 * x.
 * @param   counts  matrix of size (p.size(), x.cols()) where (i,j) entry is
 *                  the number of values of x[,j] < p[i].
 */
template <class XType, class PType, class DestType>
inline void accum_count(const XType& x, const PType& p, DestType&& counts) {
    for (int i = 0; i < x.cols(); ++i) {
        auto x_i = x.col(i);
        auto counts_i = counts.col(i);

        auto begin = x_i.data();
        auto end = begin + x_i.size();
        size_t prev_count = 0;
        int j = 0;
        while (j < p.size()) {
            auto it = std::lower_bound(begin, end, p[j]);
            if (it == end) break;
            auto n_x_less_than_pj = std::distance(begin, it) + prev_count;
            counts_i[j] = n_x_less_than_pj;
            ++j;
            for (; (j < p.size()) && (p[j] < *it); ++j) {
                counts_i[j] = n_x_less_than_pj;
            }
            begin = it;
            prev_count = n_x_less_than_pj;
        }

        // fill the rest of the counts
        if (j < p.size()) {
            counts_i.tail(p.size() - j).array() = x_i.size();
        }
    }
}

}  // namespace imprint
