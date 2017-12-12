#ifndef DD_ILP_EXTERNAL_SOLVER_INTERFACE_HXX
#define DD_ILP_EXTERNAL_SOLVER_INTERFACE_HXX

#include <type_traits>
#include <vector>
#include <limits>
#include <tuple>
#include <utility>
#include <cassert>

namespace DD_ILP {

/*
// interface for external solvers
// base class must support the following functions:

class base_solver {
public:
  // datatypes: they should be lightweight wrappers for accessing variables of the underlying solver.
  struct variable;
  struct vector;
  struct matrix; // must provide slice1 and slice2 method for obtaining matrix column and rowwise slices
  struct tensor; // must provide slice12, slice13 and slice23 for obtaining corresponding slices of tensor

  variable add_variable();
  vector add_vector(const std::size_t s);
  matrix add_matrix(const std::size_t dim1, const std::size_t dim2);
  tensor add_tensor(const std::size_t dim1, const std::size_t dim2, const std::size_t dim3);

  template<typename T>
  void add_objective(variable i, const T x);

  // constraint functions
  void add_implication(const variable i, const variable j);
  template<typename ITERATOR>
  void add_at_most_one_constraint(ITERATOR var_begin, ITERATOR var_end);
  template<typename ITERATOR>
  void add_simplex_constraint(ITERATOR var_begin, ITERATOR var_end);
  void make_equal(const variable i, const variable j) // optional, can be emulated via add_implication

  // solution functions
  void solve();
  bool solution(variable i);
};
*/

struct variable_counters {
  std::size_t variables_counter;
  std::size_t vectors_counter;
  std::size_t matrices_counter;
  std::size_t tensors_counter;
};

template<typename BASE_SOLVER>
class external_solver_interface : public BASE_SOLVER {
public:

  // solver must make available the following datatypes:
  // (i) variable
  // (ii) vector
  // (iii) matrix
  // (iv) tensor (3-dimensional)
  // the above datatypes must have a constructor which takes a REAL, LP_MP::vector, LP_MP::matrix or LP_MP::tensor respectively for initializing their size and objective

  using variable = typename BASE_SOLVER::variable;
  using vector = typename BASE_SOLVER::vector;
  using matrix = typename BASE_SOLVER::matrix;
  using tensor = typename BASE_SOLVER::tensor;

  // helper functions for converting solutions
  template<typename ITERATOR>
  std::size_t no_active(ITERATOR begin, ITERATOR end) const
  {
    std::size_t i=0;
    for(auto it=begin; it!=end; ++it) {
        if(static_cast<const BASE_SOLVER*>(this)->solution(*it) == true) {
          i++;
        }
     }
     return i; 
  }

  template<typename ITERATOR>
  std::size_t first_active(ITERATOR begin, ITERATOR end) const
  {
    std::size_t i=0;
    for(auto it=begin; it!=end; ++it, ++i) {
        if(static_cast<const BASE_SOLVER*>(this)->solution(*it) == true) {
           return i;
        }
     }
     assert(false);
     return std::numeric_limits<std::size_t>::max();
  }

  std::size_t first_active(const vector& v) const
  {
    return first_active(v.begin(), v.end());
  }

  std::array<std::size_t,2> first_active(const matrix& m) const
  {
     for(std::size_t i=0; i<m.dim1(); ++i) {
        for(std::size_t j=0; j<m.dim2(); ++j) {
           if(static_cast<const BASE_SOLVER*>(this)->solution(m(i,j)) == true) {
              return std::array<std::size_t,2>({i,j});
           }
        }
     }
     assert(false);
     return std::array<std::size_t,2>({std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()});
  }

  std::array<std::size_t,3> first_active(const tensor& t) const
  {
     for(std::size_t i=0; i<t.dim1(); ++i) {
        for(std::size_t j=0; j<t.dim2(); ++j) {
           for(std::size_t h=0; h<t.dim3(); ++h) {
              if(static_cast<const BASE_SOLVER*>(this)->solution(t(i,j,h)) == true) {
                 return std::array<std::size_t,3>({i,j,h});
              }
           }
        }
     }
     assert(false);
     return std::array<std::size_t,3>({std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()});
  }

  // mechanism to obtain variables: In constraint construction variables are added and can afterwards be retrieved via the corresponding load_...-functions.
  // when adding variables via the methods that work
  variable add_variable() 
  { 
    auto var = static_cast<BASE_SOLVER*>(this)->add_variable(); 
    variables_.push_back(var);
    return var;
  }
  template<typename T>
  variable add_variable(const T val)
  {
    return add_variable(); 
  }

  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value,vector>::type add_vector(const T size) 
  { 
    auto vec = static_cast<BASE_SOLVER*>(this)->add_vector(size); 
    vectors_.push_back(vec);
    return vec;
  }
  template<typename VECTOR>
  typename std::enable_if<!std::is_arithmetic<VECTOR>::value,vector>::type add_vector(const VECTOR& v) 
  { 
    return add_vector(v.size()); 
  }

  matrix add_matrix(const std::size_t dim1, const std::size_t dim2) 
  { 
    auto mat = static_cast<BASE_SOLVER*>(this)->add_matrix(dim1, dim2);
    matrices_.push_back(mat);
    return mat;
  }
  template<typename MATRIX>
  matrix add_matrix(const MATRIX& m) 
  { return add_matrix(m.dim1(), m.dim2()); }

  tensor add_tensor(const std::size_t dim1, const std::size_t dim2, const std::size_t dim3) 
  { 
    auto tensor = static_cast<BASE_SOLVER*>(this)->add_tensor(dim1, dim2, dim3);
    tensors_.push_back(tensor);
    return tensor;
  }
  template<typename TENSOR>
  tensor add_tensor(const TENSOR& t) 
  { return add_tensor(t.dim1(), t.dim2(), t.dim3()); } 

  // associate linear objective function to variables
  template<typename T>
  void add_objective(variable x, const T val)
  {
    static_assert(std::is_arithmetic<T>::value,"objective value for optimization problem must be a number");
    static_cast<BASE_SOLVER*>(this)->add_objective(x, val);
  }

  template<typename T>
  void add_variable_objective(const T val)
  {
    auto var = load_variable();
    add_objective(val, val);
  }
  template<typename VECTOR>
  void add_vector_objective(const VECTOR& vals)
  {
    auto vec = load_vector();
    auto vec_it = vec.begin();
    auto val_it = vals.begin();
    assert(vec.size() == vals.size());
    for(; val_it!=vals.end(); ++vec_it, ++val_it) {
      add_objective(*vec_it, *val_it);
    }
  }
  template<typename MATRIX>
  void add_matrix_objective(const MATRIX& vals)
  {
    auto mat = load_matrix();
    assert(mat.dim1() == vals.dim1() && mat.dim2() == vals.dim2());
    for(std::size_t i=0; i<mat.dim1(); ++i) {
      for(std::size_t j=0; j<mat.dim2(); ++j) {
        add_objective(mat(i,j), vals(i,j));
      }
    }
  }
  template<typename TENSOR>
  void add_tensor_objective(const TENSOR& vals)
  {
    auto t = load_tensor();
    assert(t.dim1() == vals.dim1() && t.dim2() == vals.dim2() && t.dim3() == vals.dim3());
    for(std::size_t i=0; i<t.dim1(); ++i) {
      for(std::size_t j=0; j<t.dim2(); ++j) {
        for(std::size_t k=0; k<t.dim3(); ++k) {
          add_objective(t(i,j,k), vals(i,j,k));
        }
      }
    }
  }

  void init_variable_loading()
  {
    variables_counter = 0;
    vectors_counter = 0;
    matrices_counter = 0;
    tensors_counter = 0;
  }

  variable load_variable() 
  { 
    assert(variables_counter < variables_.size());
    return variables_[variables_counter++];
  }
  vector load_vector() 
  { 
    assert(vectors_counter < vectors_.size());
    return vectors_[vectors_counter++];
  }
  matrix load_matrix() 
  { 
    assert(matrices_counter < matrices_.size());
    return matrices_[matrices_counter++];
  }
  tensor load_tensor()
  { 
    assert(tensors_counter < tensors_.size());
    return tensors_[tensors_counter++];
  }

  void add_implication(const variable i, const variable j) { return static_cast<BASE_SOLVER*>(this)->add_implication(i,j); }
  // potentially, if base class does not have this method, implement it in terms of add_implication

  void make_equal(const variable i, const variable j)
  {
    static_cast<BASE_SOLVER*>(this)->make_equal(i, j);
  }

  template<typename ITERATOR_1, typename ITERATOR_2>
  void make_equal(ITERATOR_1 begin_1, ITERATOR_1 end_1, ITERATOR_2 begin_2, ITERATOR_2 end_2)
  {
     assert(std::distance(begin_1, end_1) == std::distance(begin_2, end_2));
     auto it_1 = begin_1;
     auto it_2 = begin_2;
     for(; it_1!=end_1; ++it_1, ++it_2) {
        static_cast<BASE_SOLVER*>(this)->make_equal(*it_1, *it_2);
     }
     assert(it_2 == end_2);
  }

  template<typename ITERATOR>
  void add_simplex_constraint(ITERATOR var_begin, ITERATOR var_end) 
  { 
    return static_cast<BASE_SOLVER*>(this)->add_simplex_constraint(var_begin, var_end); 
  }
  template<typename ITERATOR>
  variable add_at_most_one_constraint(ITERATOR var_begin, ITERATOR var_end) 
  { 
    return static_cast<BASE_SOLVER*>(this)->add_at_most_one_constraint(var_begin, var_end); 
  }
  
  variable max(const variable i, const variable j)
  {
    std::array<variable,2> vars({i,j});
    return static_cast<BASE_SOLVER*>(this)->max(vars.begin(), vars.end()); 
  }
  template<typename ITERATOR>
  variable max(ITERATOR var_begin, ITERATOR var_end)
  {
    return static_cast<BASE_SOLVER*>(this)->max(var_begin, var_end); 
  }

  bool solve() 
  { 
    return static_cast<BASE_SOLVER*>(this)->solve(); 
  }
  bool solution(variable i) { return static_cast<BASE_SOLVER*>(this)->solution(i); }

  variable_counters get_variable_counters() const
  {
    return variable_counters({variables_.size(), vectors_.size(), matrices_.size(), tensors_.size()});
  }

  void set_variable_counters(const variable_counters& c)
  { 
    variables_counter = c.variables_counter;
    vectors_counter = c.vectors_counter;
    matrices_counter = c.matrices_counter;
    tensors_counter = c.tensors_counter;
  }

//private:
  // each factor adds new variables. We store them here so they can be again recovered later for reconstructing primal solution or initializing optimization costs;
  std::vector<variable> variables_;
  std::size_t variables_counter = 0;
  
  std::vector<vector> vectors_;
  std::size_t vectors_counter = 0;
  
  std::vector<matrix> matrices_;
  std::size_t matrices_counter = 0;

  std::vector<tensor> tensors_;
  std::size_t tensors_counter = 0;

  // Models can be reduced by exclusing variables which carry too high cost. Reduction is done automatically for the implemented constraint functions.
  struct assumption { variable var; unsigned char value; };
  std::vector<assumption> assumptions_;
};

// when reducing models, we overload the add_..._constraint methods and redirect them to reduce_..._constraint methods that push assumptions instead of constructing the actual constraints
template<typename EXTERNAL_SOLVER>
class external_solver_interface_reduction : public EXTERNAL_SOLVER {
public:
  using base_solver_type = typename EXTERNAL_SOLVER::base_solver_type;
  using variable = typename EXTERNAL_SOLVER::variable;
  using vector = typename EXTERNAL_SOLVER::vector;

  template<typename ITERATOR>
  void add_simplex_constraint(ITERATOR var_begin, ITERATOR var_end) 
  { 
    // get values associated to variables
    assert(false);
  }

  struct variable_value_pair { variable var; double val; };
  std::vector<variable_value_pair> values;

  //struct vector_values_pair { vector vec; };
};


} // end namespace DD_ILP

#endif // DD_ILP_EXTERNAL_SOLVER_INTERFACE_HXX
