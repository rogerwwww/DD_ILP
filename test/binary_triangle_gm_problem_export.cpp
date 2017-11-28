#include "binary_triangle_gm.hxx"
#include "include/external_solver_interface.hxx"
#include "include/problem_export.hxx"

using namespace DD_ILP;

int main()
{
  auto* s = build_triangle_gm<problem_export>();
  s->write_to_file("binary_triangle_gm.lp");
  delete s;
}

