#include <gtest/gtest.h>

#include <imprint_bits/util/progress_bar.hpp>

namespace imprint {

struct pb_fixture : ::testing::Test {
   protected:
};

TEST_F(pb_fixture, ctor) { ProgressBar pb(10); }

TEST_F(pb_fixture, update_test) {
    int n = 10000;
    ProgressBar pb(n);
    for (int i = 0; i < n; ++i) {
        pb.update(std::cout);
    }
}

TEST_F(pb_fixture, update_test_bar_length) {
    int n = 10000;
    ProgressBar pb(n, 38);
    for (int i = 0; i < n; ++i) {
        pb.update(std::cout);
    }
}

}  // namespace imprint
