#include <gtest/gtest.h>

#include <imprint_bits/util/types.hpp>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace imprint {

struct types_fixture : ::testing::Test {
   protected:
    template <class TrueTab, class Comp>
    void check_op(const TrueTab& true_tab, Comp comp) {
        for (orient_type it1 = orient_type::begin; it1 != orient_type::end;
             ++it1) {
            for (orient_type it2 = orient_type::begin; it2 != orient_type::end;
                 ++it2) {
                if ((true_tab.find(it1) != true_tab.end()) &&
                    (true_tab.at(it1).find(it2) != true_tab.at(it1).end())) {
                    EXPECT_TRUE(comp(it1, it2));
                } else {
                    EXPECT_FALSE(comp(it1, it2));
                }
            }
        }
    }
};

TEST_F(types_fixture, orient_type_le) {
    const std::unordered_map<orient_type, std::unordered_set<orient_type>>
        true_tab = {
            {orient_type::pos, {orient_type::non_neg, orient_type::non_on}},
            {orient_type::on, {orient_type::non_neg, orient_type::non_pos}},
            {orient_type::neg, {orient_type::non_pos, orient_type::non_on}},
        };

    check_op(true_tab, std::less<orient_type>());
}

TEST_F(types_fixture, orient_type_leq) {
    const std::unordered_map<orient_type, std::unordered_set<orient_type>>
        true_tab = {
            {orient_type::pos,
             {orient_type::non_neg, orient_type::non_on, orient_type::pos}},
            {orient_type::on,
             {orient_type::non_neg, orient_type::non_pos, orient_type::on}},
            {orient_type::neg,
             {orient_type::non_pos, orient_type::non_on, orient_type::neg}},
            {orient_type::non_pos, {orient_type::non_pos}},
            {orient_type::non_neg, {orient_type::non_neg}},
            {orient_type::non_on, {orient_type::non_on}},
            {orient_type::none, {orient_type::none}},
        };

    check_op(true_tab, std::less_equal<orient_type>());
}

}  // namespace imprint
