#include <stdexcept>
#include <string>

inline void test(const bool& pred)
{
  if (!pred)
    throw std::runtime_error("Test failed.");
}
