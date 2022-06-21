#pragma once
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace imprint {

struct ProgressBar {
    ProgressBar(ProgressBar const&) = delete;
    ProgressBar& operator=(ProgressBar const&) = delete;
    ProgressBar(ProgressBar&&) = delete;
    ProgressBar& operator=(ProgressBar&&) = delete;

    ProgressBar() : ProgressBar(0) {}

    ProgressBar(int n, int bar_length = 50, bool show_time = true)
        : n_total_(n),
          bar_length_(bar_length),
          incr_size_(100. / bar_length_),
          show_time_(show_time) {}

    void reset() {
        prev_perc_ = 0;
        n_finished_ = 0;
        n_iter_ = 0;
        bar_created_ = false;
    }

    void set_n_total(int n) {
        check_n_(n);
        n_total_ = n;
    }

    void set_finish_char(char c) { finish_char_ = c; }
    void set_remain_char(char c) { remain_char_ = c; }
    void set_begin_char(char c) { begin_char_ = c; }
    void set_end_char(char c) { end_char_ = c; }
    void set_show_time(bool b) { show_time_ = b; }

    template <class OStreamType>
    void initialize(OStreamType& os) {
        // create the bar if not created yet
        if (!bar_created_) {
            os << begin_char_;
            for (int i = 0; i < bar_length_; ++i) {
                os << remain_char_;
            }
            os << end_char_ << " 0% ";
            if (show_time_) os << "[00:00:00]";
            os.flush();
            bar_created_ = true;
            tp_ = clck_t::now();
        }
    }

    template <class OStreamType>
    void finish(OStreamType& os) {
        n_iter_ = n_total_ - 1;
        update(os);
    }

    template <class OStreamType>
    void update(OStreamType& os, size_t amt = 1) {
        // check that n is positive
        check_n_(n_total_);

        auto new_tp = clck_t::now();

        // create bar if wasn't created already
        if (!bar_created_) initialize(os);

        // increase number of iterations
        n_iter_ += amt;

        double perc =
            (n_iter_ == n_total_) ? 100. : (n_iter_ * 100.) / n_total_;
        int new_n_finished = perc / incr_size_;
        int extra_n_finished = new_n_finished - n_finished_;

        // delete space + [ + ] + the time characters
        if (show_time_) os << "\b\b\b\b\b\b\b\b\b\b\b";

        // delete the space and percentage sign
        os << "\b\b";

        // delete the actual percentage number
        if (prev_perc_ < 10)
            os << "\b";
        else if (prev_perc_ >= 10 && prev_perc_ < 100)
            os << "\b\b";
        else if (prev_perc_ == 100)
            os << "\b\b\b";

        // Update the bar if necessary
        if (extra_n_finished > 0) {
            // delete end_char_
            os << '\b';

            // delete some remaining chars, add extra finish chars, add back
            // remaining chars
            for (int j = 0; j < bar_length_ - n_finished_; ++j) os << '\b';
            for (int j = 0; j < extra_n_finished; ++j) os << finish_char_;
            for (int j = 0; j < bar_length_ - new_n_finished; ++j)
                os << remain_char_;

            // add back end_char_
            os << end_char_;
        }

        os << ' ' << static_cast<int>(perc) << '%';

        if (show_time_) {
            os << " [";

            size_t dur =
                std::chrono::duration_cast<std::chrono::seconds>(new_tp - tp_)
                    .count();
            format_time(dur, os);

            os << ']';
        }

        // add a newline at the last iteration
        if (n_iter_ == n_total_) os << '\n';

        // leave invariants
        prev_perc_ = perc;
        n_finished_ = new_n_finished;

        os.flush();
    }

   private:
    using clck_t = std::chrono::steady_clock;

    static void check_n_(int n) {
        if (n <= 0)
            throw std::invalid_argument(
                "Number of iterations must be positive.");
    }

    template <class OStreamType>
    static void format_time(size_t dur, OStreamType& os) {
        int hours = (dur / 3600) % 24;
        dur %= 3600;
        int minutes = dur / 60;
        dur %= 60;
        int seconds = dur;

        auto formatter = [&](int x) {
            if (x < 10)
                os << '0' << x;
            else if (x < 100)
                os << x;
        };

        formatter(hours);
        os << ':';
        formatter(minutes);
        os << ':';
        formatter(seconds);
    }

    // configurable only at construction
    int n_total_;
    int bar_length_;

    // invariant
    int prev_perc_ = 0;
    int n_finished_ = 0;
    int n_iter_ = 0;
    bool bar_created_ = false;

    // configurable dynamically
    unsigned char finish_char_ = '=';
    char remain_char_ = ' ';
    char begin_char_ = '[';
    char end_char_ = ']';

    double incr_size_;
    bool show_time_;

    std::chrono::time_point<clck_t> tp_;
};

template <class OStreamType>
struct ProgressBarOSWrapper : ProgressBar {
    using base_t = ProgressBar;
    using base_t::base_t;

    ProgressBarOSWrapper(OStreamType& os) : os_(os) {}

    void initialize() { base_t::initialize(os_); }
    void finish() { base_t::finish(os_); }
    void update(size_t amt = 1) { base_t::update(os_, amt); }

   private:
    using base_t::finish;
    using base_t::initialize;
    using base_t::update;

    OStreamType& os_;
};

// add deduction guide
template <class OStreamType>
ProgressBarOSWrapper(OStreamType&)
    -> ProgressBarOSWrapper<std::decay_t<OStreamType> >;

// helper alias for progress bar that wraps an std::ostream
using pb_ostream = ProgressBarOSWrapper<std::ostream>;

// Dummy ostream used to suck in all inputs.
// Constructing a ProgressBarOSWrapper with void_ostream will nullify all
// progress bar operations.
struct void_ostream {};

template <>
struct ProgressBarOSWrapper<void_ostream> {
    ProgressBarOSWrapper() = default;

    // Just to keep same interface as primary definition.
    template <class T>
    ProgressBarOSWrapper(T&) {}

    void reset() {}
    void set_n_total(int) {}
    void set_finish_char(char) {}
    void set_remain_char(char) {}
    void set_begin_char(char) {}
    void set_end_char(char) {}
    void set_show_time(bool) {}
    void initialize() {}
    void finish() {}
    void update(size_t = 1) {}
};

}  // namespace imprint
