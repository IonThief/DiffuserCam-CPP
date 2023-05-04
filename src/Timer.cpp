#include "Timer.hpp"


using namespace std;
using namespace std::chrono;


Timer::Timer(vector<microseconds> *durations)
: m_durations(durations)
{
  /* store counter start value */
  startTimePoint = high_resolution_clock::now();
}


Timer::~Timer(){
  /* store counter stop value */
  auto endTimePoint = high_resolution_clock::now();

  /* retrieve duration */
  auto start = time_point_cast<microseconds>(startTimePoint).time_since_epoch().count();
  auto end   = time_point_cast<microseconds>(endTimePoint).time_since_epoch().count();
  auto duration = end - start;

  /* store measurement for later use */
  m_durations->push_back(microseconds(duration));
}
