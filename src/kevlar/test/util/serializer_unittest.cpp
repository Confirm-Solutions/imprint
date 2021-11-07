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
    // Brace is to make sure it destructs before TearDown.
    // TearDown needs to remove the tmp file and these objects must release the resource first.
    {
        Serializer s(fname);
    }
}

struct serializer_fixture_param
    : serializer_fixture,
      testing::WithParamInterface<
        std::tuple<size_t> >
{
protected:
    size_t n;
    Eigen::VectorXd x;

    serializer_fixture_param()
    {
        std::tie(n) = GetParam();
        x.setRandom(n);
    }
};

TEST_P(serializer_fixture_param, serializer_op_double_put_1)
{
    {
        double expected = 5.;
        Serializer s(fname);
        s << expected; 

        UnSerializer us(fname);         
        double actual;
        us >> actual;

        EXPECT_DOUBLE_EQ(actual, expected);
    }
}

TEST_P(serializer_fixture_param, serializer_op_double_put_5)
{
    {
        std::array<double, 5> expected = {1, 2, 3, 4, 5};
        Serializer s(fname);
        for (int i = 0; i < 5; ++i) {
            s << expected[i];
        }

        UnSerializer us(fname);
        
        for (int i = 0; i < 5; ++i) {
            double actual;
            us >> actual;
            EXPECT_DOUBLE_EQ(actual, expected[i]);
        }
    }
}

TEST_P(serializer_fixture_param, serializer_op_vector_put_1)
{
    {
        Serializer s(fname);
        s << x; 

        UnSerializer us(fname);         

        Eigen::VectorXd actual(n);
        us >> actual;

        expect_double_eq_vec(actual, x);
    }
}

TEST_P(serializer_fixture_param, serializer_op_vector_put_5)
{
    {
        Serializer s(fname);
        s << x; 
        s << x; 
        s << x; 
        s << x; 
        s << x; 

        UnSerializer us(fname);         
        for (int i = 0; i < 5; ++i) {
            Eigen::VectorXd actual(n);
            us >> actual;
            expect_double_eq_vec(actual, x);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    SerializerSuite, serializer_fixture_param,

    testing::Combine(
        testing::Values(1, 2, 3, 5, 10, 15, 20, 100)
        )
);

} // namespace kevlar
