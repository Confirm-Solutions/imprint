#include <testutil/base_fixture.hpp>
#include <kevlar_bits/util/serializer.hpp>
#include <filesystem>

namespace kevlar {

struct serializer_fixture
    : base_fixture
{
    void TearDown() override 
    {
        std::filesystem::path p(fname);
        bool status = std::filesystem::remove(p); 
        EXPECT_TRUE(status);
    }

protected:
    static constexpr const char* fname = "tmp";
};

TEST_F(serializer_fixture, serializer_ctor)
{
    Serializer s(fname);
}

struct serializer_fixture_param
    : serializer_fixture,
      testing::WithParamInterface<
        std::tuple<size_t> >
{
protected:
    size_t n;
    Eigen::VectorXd x;
    Serializer s;

    serializer_fixture_param()
        : s(fname)
    {
        std::tie(n) = GetParam();
        x.setRandom(n);
    }

    Eigen::VectorXd read_file(size_t size)
    {
        Eigen::VectorXd x(size);
        UnSerializer us(fname);         
        us >> x;
        return x;
    }
};

TEST_P(serializer_fixture_param, serializer_op_put_1)
{
    s << x; 
    Eigen::VectorXd actual = read_file(n).col(0);
    expect_double_eq_vec(actual, x);
}

TEST_P(serializer_fixture_param, serializer_op_put_5)
{
    s << x; 
    s << x; 
    s << x; 
    s << x; 
    s << x; 

    Eigen::VectorXd actual = read_file(5 * n);
    Eigen::Map<Eigen::MatrixXd> actual_map(actual.data(), n, 5);
    for (int i = 0; i < actual_map.cols(); ++i) {
        auto actual_i = actual_map.col(i);
        expect_double_eq_vec(actual_i, x);
    }
}

INSTANTIATE_TEST_SUITE_P(
    SerializerSuite, serializer_fixture_param,

    testing::Combine(
        testing::Values(1, 2, 3, 5, 10, 15, 20, 100)
        )
);

} // namespace kevlar
