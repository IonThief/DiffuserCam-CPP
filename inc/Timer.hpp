#ifndef _TIMER_HPP_
#define _TIMER_HPP_

#include <vector>
#include <chrono>

class Timer{
  private:
    std::vector<std::chrono::microseconds> *m_durations;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTimePoint;

  public:
    Timer(std::vector<std::chrono::microseconds> *durations);
    ~Timer();
};

#endif
