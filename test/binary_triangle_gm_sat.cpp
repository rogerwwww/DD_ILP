#include "binary_triangle_gm.hxx"
#include "DD_ILP.hxx"

using namespace DD_ILP;

int main()
{
  auto* s = build_triangle_gm<sat_solver>();
  test_triangle_gm(s);
  delete s;
}
