#ifndef DD_ILP_TEST_BINARY_TRIANGLE_GM_HXX
#define DD_ILP_TEST_BINARY_TRIANGLE_GM_HXX

#include "test.h"
#include "DD_ILP.hxx"

template<typename SOLVER, typename ITERATOR>
bool one_hot_variable(SOLVER& s, ITERATOR begin, ITERATOR end)
{
  std::size_t no_active_variables = 0;
  for(auto it=begin; it!=end; ++it) {
    if(s.solution(*it)) { no_active_variables++; }
  }
  return no_active_variables == 1;
}

// construct binary graphical model on triangle that is not LP-tight for the local polytope relaxation
template<typename SOLVER>
DD_ILP::external_solver_interface<SOLVER>* build_triangle_gm()
{
  DD_ILP::external_solver_interface<SOLVER>* s = new DD_ILP::external_solver_interface<SOLVER>();

  {
    // construct problem
    auto unary1 = s->add_vector(2);
    auto unary2 = s->add_vector(2);
    auto unary3 = s->add_vector(2);

    s->add_simplex_constraint(unary1.begin(), unary1.end());
    s->add_simplex_constraint(unary2.begin(), unary2.end());
    s->add_simplex_constraint(unary3.begin(), unary3.end());

    auto pairwise12 = s->add_matrix(2,2);
    auto pairwise13 = s->add_matrix(2,2);
    auto pairwise23 = s->add_matrix(2,2);

    s->make_equal(unary1[0], s->max(pairwise12(0,0), pairwise12(0,1)));
    s->make_equal(unary1[1], s->max(pairwise12(1,0), pairwise12(1,1)));
    s->make_equal(unary2[0], s->max(pairwise12(0,0), pairwise12(1,0)));
    s->make_equal(unary2[1], s->max(pairwise12(0,1), pairwise12(1,1)));

    s->make_equal(unary1[0], s->max(pairwise13(0,0), pairwise13(0,1)));
    s->make_equal(unary1[1], s->max(pairwise13(1,0), pairwise13(1,1)));
    s->make_equal(unary3[0], s->max(pairwise13(0,0), pairwise13(1,0)));
    s->make_equal(unary3[1], s->max(pairwise13(0,1), pairwise13(1,1)));

    s->make_equal(unary2[0], s->max(pairwise23(0,0), pairwise23(0,1)));
    s->make_equal(unary2[1], s->max(pairwise23(1,0), pairwise23(1,1)));
    s->make_equal(unary3[0], s->max(pairwise23(0,0), pairwise23(1,0)));
    s->make_equal(unary3[1], s->max(pairwise23(0,1), pairwise23(1,1)));
  }

  return s;
}

template<typename SOLVER>
void test_triangle_gm(DD_ILP::external_solver_interface<SOLVER>* s)
{

  s->solve();
  {
    // load variables
    auto unary1 = s->load_vector();
    test(unary1.size() == 2);
    auto unary2 = s->load_vector();
    test(unary2.size() == 2);
    auto unary3 = s->load_vector();
    test(unary3.size() == 2);

    auto pairwise12 = s->load_matrix();
    test(pairwise12.dim1() == 2 && pairwise12.dim2() == 2);
    auto pairwise13 = s->load_matrix();
    test(pairwise13.dim1() == 2 && pairwise13.dim2() == 2);
    auto pairwise23 = s->load_matrix();
    test(pairwise23.dim1() == 2 && pairwise23.dim2() == 2);

    // check solution for feasibility
    test(one_hot_variable(*s, unary1.begin(), unary1.end()));
    test(one_hot_variable(*s, unary2.begin(), unary2.end()));
    test(one_hot_variable(*s, unary3.begin(), unary3.end()));
    test(one_hot_variable(*s, pairwise12.begin(), pairwise12.end()));
    test(one_hot_variable(*s, pairwise13.begin(), pairwise13.end()));
    test(one_hot_variable(*s, pairwise23.begin(), pairwise23.end()));

    std::size_t x1 = s->first_active(unary1);
    test(x1<2);
    std::size_t x2 = s->first_active(unary2);
    test(x2<2);
    std::size_t x3 = s->first_active(unary3);
    test(x3<2);

    test(s->solution(pairwise12(x1,x2)));
    test(s->solution(pairwise13(x1,x3)));
    test(s->solution(pairwise23(x2,x3)));
  }
}


#endif // DD_ILP_TEST_BINARY_TRIANGLE_GM_HXX
