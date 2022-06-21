#pragma once
#include <exception>
#include <string>

namespace imprint {

struct imprint_error : std::exception {};

struct min_lmda_reached_error : imprint_error {
    min_lmda_reached_error()
        : msg_("Min lmda reached. Try a grid of lambda with lower values.") {}

    const char* what() const noexcept override { return msg_.data(); }

   private:
    std::string msg_;
};

}  // namespace imprint
