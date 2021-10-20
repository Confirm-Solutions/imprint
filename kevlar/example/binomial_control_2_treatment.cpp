#include <Eigen/Core>
#include <random>

template <class PType>
void run(const PType& p, size_t seed)
{
    constexpr size_t phase_2_size = 50;

    std::mt19937 gen(seed);
    std::uniform_real_distribution<> unif;
    
    // Phase II


    // Compare and choose arm with more successes
    bool choose_a1 = a1.sum() >= a2.sum();

}


int main()
{
    return 0;
}
