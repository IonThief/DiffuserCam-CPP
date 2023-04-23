#ifndef __CLASSES_HPP__
#define __CLASSES_HPP__

#include <iostream>
#include <fstream>
#include <chrono>
#define CSV_FILE "time.csv"


class csvfile{
	std::ofstream fs_;
	const std::string separator_;

	public:
	  csvfile(const std::string filename, const std::string separator = ",") : fs_(), separator_(separator){
	    fs_.exceptions(std::ios::failbit | std::ios::badbit);
	    fs_.open(filename, std::ios::app);
	  }

	  ~csvfile(){
	    flush();
	    fs_.close();
	  }

	  void flush(){
	    fs_.flush();
	  }

	  void endrow(){
	    fs_ << std::endl;
	  }

	  csvfile& operator << ( csvfile& (* val)(csvfile&)){
	    return val(*this);
	  }

	  csvfile& operator << (const char * val){
	    fs_ << '"' << val << '"' << separator_;
	    return *this;
	  }

	  csvfile& operator << (const std::string & val){
	    fs_ << '"' << val << '"' << separator_;
	    return *this;
	  }

	  template<typename T>
	  csvfile& operator << (const T& val){
	    fs_ << val << separator_;
	    return *this;
	  }
};

inline static csvfile& endrow(csvfile& file){
  file.endrow();
  return file;
}

inline static csvfile& flush(csvfile& file){
  file.flush();
  return file;
}


class Timer{
	private:
		std::vector<std::chrono::microseconds> *m_durations;
		std::chrono::time_point<std::chrono::high_resolution_clock> startTimePoint;

	public:
		Timer(std::vector<std::chrono::microseconds> *durations) : m_durations(durations){
			startTimePoint = std::chrono::high_resolution_clock::now();
		}
		~Timer(){
			auto endTimePoint = std::chrono::high_resolution_clock::now();
			auto start = std::chrono::time_point_cast<std::chrono::microseconds>(startTimePoint).time_since_epoch().count();
			auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimePoint).time_since_epoch().count();
			auto duration = end - start;
			m_durations->push_back(std::chrono::microseconds(duration));
		}
};

#endif
