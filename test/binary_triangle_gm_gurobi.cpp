#include "binary_triangle_gm.hxx"
#include "gurobi_interface.hxx"

using namespace DD_ILP;

int main()
{
  auto* s = build_triangle_gm<gurobi_interface>();
  s->write_to_file("binary_triangle_gm_gurobi.lp");
  test_triangle_gm(s);
  delete s;
}


