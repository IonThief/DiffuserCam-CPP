#include "Csvfile.hpp"


Csvfile::Csvfile(const std::string filename, const std::string separator)
: fs_()
, separator_(separator)
{
  fs_.exceptions(std::ios::failbit | std::ios::badbit);
  fs_.open(filename, std::ios::app);
}

Csvfile::~Csvfile(){
  flush();
  fs_.close();
}

void Csvfile::flush(){
  fs_.flush();
}

void Csvfile::endrow(){
  fs_ << std::endl;
}
