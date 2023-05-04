#ifndef _CSVFILE_HPP_
#define _CSVFILE_HPP_

#include <iostream>
#include <fstream>
#define CSV_FILE "time.csv"


class Csvfile{
  private:
    std::ofstream fs_;
    const std::string separator_;

  public:
    Csvfile(const std::string filename, const std::string separator = ",");
    ~Csvfile();

    void flush();
    void endrow();

    Csvfile& operator << ( Csvfile& (* val)(Csvfile&)){
      return val(*this);
    }

    Csvfile& operator << (const char * val){
      fs_ << val << separator_;
      return *this;
    }

    Csvfile& operator << (const std::string & val){
      fs_ << val << separator_;
      return *this;
    }

    template<typename T>
    Csvfile& operator << (const T& val){
      fs_ << val << separator_;
      return *this;
    }
};

inline static Csvfile& endrow(Csvfile& file){
  file.endrow();
  return file;
}

inline static Csvfile& flush(Csvfile& file){
  file.flush();
  return file;
}


#endif
