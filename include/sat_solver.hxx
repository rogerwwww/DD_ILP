#ifndef DD_ILP_SAT_SOLVER_HXX
#define DD_ILP_SAT_SOLVER_HXX

#include <vector>
#include <algorithm>

namespace DD_ILP {

extern "C" {
#include "lglib.h"
}

// wraps the lingeling sat solver and adds functions for adding constraints
// solver directly works on literals, not variables
// possibly implement this as separate project.
// also templatize w.r.t. underlying sat solver. Expect picosat like interface.
// In particular, the sat_opt_solver class needs picosat, sat_solver however can also use lingeling

class sat_solver {
// datatypes for holding sat variables
public:
using sat_var = int;
using sat_literal = int;
using sat_vec = std::vector<sat_literal>;

struct sat_literal_vector_iterator : std::iterator< std::random_access_iterator_tag, sat_literal >{
   sat_literal_vector_iterator(sat_literal _l) : l(_l) {}
   bool operator==(const sat_literal_vector_iterator o) const { return l == o.l; }
   bool operator!=(const sat_literal_vector_iterator o) const { return !(*this == o); }
   void operator++() { ++l; }
   const sat_literal operator*() const { return l; }
   int operator-(sat_literal_vector_iterator o) const { return l - o.l; }
   sat_literal_vector_iterator operator+(const int n) const { return sat_literal_vector_iterator(l + n); }
   sat_literal_vector_iterator operator-(const int n) const { return sat_literal_vector_iterator(l - n); }
   private:
   sat_literal l;
};

struct sat_literal_vector_iterator_strided : std::iterator< std::random_access_iterator_tag, sat_literal >{
   sat_literal_vector_iterator_strided(sat_literal _l, const std::size_t _stride) : l(_l), stride(_stride) {}
   bool operator==(const sat_literal_vector_iterator_strided o) const { return (l == o.l && stride == o.stride); }
   bool operator!=(const sat_literal_vector_iterator_strided o) const { return !(*this == o); }
   void operator++() { l+=stride; }
   const sat_literal operator*() const { return l; }
   int operator-(sat_literal_vector_iterator_strided o) const { return (l - o.l)/stride; }
   sat_literal_vector_iterator_strided operator+(const int n) const { return sat_literal_vector_iterator_strided(l + n*stride, stride); } 
   private:
   sat_literal l;
   std::size_t stride;
};


struct sat_literal_vector {
   template<typename VECTOR>
   sat_literal_vector(const VECTOR& v) : begin_(0), dim_(v.size()) {}

   sat_literal_vector(const std::size_t dim) : begin_(0), dim_(dim) 
   {
      assert(dim > 0);
   } 
   sat_literal_vector(sat_literal f, const std::size_t dim) : begin_(f), dim_(dim) 
   {
      assert(dim > 0);
      assert(f > 0);
   } 
   void set_begin(sat_literal l)
   {
      assert(l > 0);
      begin_ = l;
   }
   bool operator==(const sat_literal_vector o) const
   {
      return begin_ == o.begin_ && size() == o.size();
   }

   std::size_t size() const { return dim_; }
   const sat_literal operator[](const std::size_t i) const { assert(i < size()); return begin_ + i; }
   const sat_literal back() const { return begin_ + size()-1; }

   auto begin() const {
      return sat_literal_vector_iterator(begin_);
   }
   auto end() const {
      return sat_literal_vector_iterator(begin_ + size());
   }
   private:
   sat_literal begin_;
   const std::size_t dim_;
};

struct sat_literal_vector_strided {
   sat_literal_vector_strided(sat_literal f, const std::size_t dim, const std::size_t stride) : begin_(f), dim_(dim), stride_(stride)
   {
      assert(dim > 0 && stride_ > 0);
      assert(f > 0);
   } 
   bool operator==(const sat_literal_vector_strided o) const
   {
      return begin_ == o.begin_ && size() == o.size() && stride_ == o.stride_;
   }

   std::size_t size() const { return dim_; }
   const sat_literal operator[](const std::size_t i) const { assert(i < size()); return begin_ + i*stride_; }
   const sat_literal back() const { return (*this)[size()-1]; }

   auto begin() const {
      return sat_literal_vector_iterator_strided(begin_, stride_);
   }
   auto end() const {
      return sat_literal_vector_iterator_strided(begin_ + stride_*size(), stride_);
   }
   private:
   sat_literal begin_;
   const std::size_t dim_;
   const std::size_t stride_;
}; 

struct sat_literal_matrix {
   template<typename MATRIX>
   sat_literal_matrix(const MATRIX& m) : begin_(0), dim_({m.dim1(), m.dim2()}) {}

   sat_literal_matrix(const std::size_t n, const std::size_t m) : begin_(0), dim_({n,m}) 
   {
      assert(n > 0 && m > 0); 
   }
   sat_literal_matrix(sat_literal f, const std::size_t n, const std::size_t m) : begin_(f), dim_({n,m}) 
   {
      assert(n > 0 && m > 0);
      assert(f > 0);
   } 
   void set_begin(sat_literal l)
   {
      assert(l > 0);
      begin_ = l;
   }
   bool operator==(const sat_literal_matrix& o) const
   {
      return begin_ == o.begin_ && dim_ == o.dim_;
   }

   std::size_t size() const { return dim_[0]*dim_[1]; }
   std::size_t dim1() const { return dim_[0]; }
   std::size_t dim2() const { return dim_[1]; }
   std::size_t dim(const std::size_t d) const { assert(d<2); return dim_[d]; }
   const sat_literal operator[](const std::size_t i) const 
   {
      assert(begin_ > 0);
      assert(i < size());
      return begin_ + i;
   }
   const sat_literal operator()(const std::size_t x1, const std::size_t x2) const 
   { 
      assert(begin_ > 0);
      assert(x1 < dim(0) && x2 < dim(1));
      return begin_ + x1*dim(1) + x2;
   }

   auto begin() const {
      return sat_literal_vector_iterator(begin_);
   }
   auto end() const {
      return sat_literal_vector_iterator(begin_ + size());
   }

   sat_literal_vector slice_left(const std::size_t x1) const {
      assert(x1 < dim(0));
      return sat_literal_vector(begin_ + x1*dim(1),dim(1));
   }
   auto slice1(const std::size_t x1) const { return slice_left(x1); }

   sat_literal_vector_strided slice_right(const std::size_t x2) const {
      assert(x2 < dim(1));
      return sat_literal_vector_strided(begin_ + x2, dim(0), dim(1));
   }
   auto slice2(const std::size_t x2) const { return slice_right(x2); }
   private:
   sat_literal begin_;
   const std::array<std::size_t,2> dim_;
};

struct sat_literal_tensor {
   template<typename TENSOR>
   sat_literal_tensor(const TENSOR& t) : begin_(0), dim_({t.dim1(), t.dim2(), t.dim3()}) {}

   sat_literal_tensor(const std::size_t n, const std::size_t m, const std::size_t k) : begin_(0), dim_({n,m,k}) 
   {
      assert(n > 0 && m > 0 && k > 0);
   } 
   sat_literal_tensor(sat_literal f, const std::size_t n, const std::size_t m, const std::size_t k) : begin_(f), dim_({n,m,k}) 
   {
      assert(n > 0 && m > 0 && k > 0);
      assert(f > 0);
   } 
   void set_begin(sat_literal l)
   {
      assert(l > 0);
      begin_ = l;
   }
   bool operator==(const sat_literal_tensor& o) const
   {
      return begin_ == o.begin_ && dim_ == o.dim_;
   }

   std::size_t size() const { return dim_[0]*dim_[1]*dim_[2]; }
   std::size_t dim(const std::size_t d) const { assert(d<3); return dim_[d]; }
   const sat_literal operator[](const std::size_t i) const 
   {
      assert(i < size());
      return begin_ + i;
   }
   const sat_literal operator()(const std::size_t x1, const std::size_t x2, const std::size_t x3) const 
   { 
      assert(x1 < dim(0) && x2 < dim(1) && x3 < dim(2));
      return begin_ + x1*dim(1)*dim(2) + x2*dim(2) + x3;
   }

   auto begin() const {
      return sat_literal_vector_iterator(begin_);
   }
   auto end() const {
      return sat_literal_vector_iterator(begin_ + size());
   }

   sat_literal_vector slice12(const std::size_t x1, const std::size_t x2) const
   {
      assert(x1 < dim(0) && x2 < dim(1));
      return sat_literal_vector(begin_ + x1*dim(2)*dim(1) + x2*dim(2),dim(2)); 
   }
   sat_literal_vector_strided slice13(const std::size_t x1, const std::size_t x3) const
   {
      assert(x1 < dim(0) && x3 < dim(2));
      return sat_literal_vector_strided(begin_ + x1*dim(2)*dim(1) + x3, dim(1), dim(2)); 
   }
   sat_literal_vector_strided slice23(const std::size_t x2, const std::size_t x3) const
   {
      assert(x2 < dim(1) && x3 < dim(2));
      return sat_literal_vector_strided(begin_ + x2*dim(2) + x3, dim(0), dim(1)*dim(2)); 
   }

   private:
   sat_literal begin_;
   const std::array<std::size_t,3> dim_;
};

// make aliases to be used by external_solver_interface
using variable = sat_literal;
using vector = sat_literal_vector;
using matrix = sat_literal_matrix;
using tensor = sat_literal_tensor; 

// the actual solver
public:

   sat_solver() : sat_(lglinit())
   {
     assert(sat_ != nullptr);
     //sat_.set_no_simplify(); // seems to make solver much faster
   }

   sat_solver(sat_solver& o) : sat_(nullptr)
   {
      assert(false);
   }

   ~sat_solver()
   {
      assert(sat_ != nullptr);
      lglrelease(sat_);
   }

std::size_t size() const
   {
      return lglmaxvar(sat_);
   }

   // utility functions
   static sat_literal to_literal(const sat_var v) 
   {
      return v+1;
   }

   bool valid_literal(const sat_literal l) const
   {
      return l != 0 && std::abs(l) <= size();
   }

   static sat_var to_var(const sat_literal l)
   {
      assert(std::abs(l) > 0);
      return std::abs(l) - 1;
   }

   sat_literal add_variable()
   {
      return lglincvar(sat_);
   }

   sat_literal_vector add_vector(const std::size_t n)
   {
      assert(n > 0);
      auto first = add_variable();
      auto last = first;
      for(std::size_t i=1; i<n; ++i) {
         last = add_variable();
      } 
      return sat_literal_vector(first,n);
   }

   sat_literal_matrix add_matrix(const std::size_t n, const std::size_t m)
   {
      assert(n > 0 && m > 0);
      auto first = add_variable();
      for(std::size_t i=1; i<n*m; ++i) {
         auto l = add_variable();
         assert(l == first+i);
      } 
      return sat_literal_matrix(first,n,m);
   }

   sat_literal_tensor add_tensor(const std::size_t n, const std::size_t m, const std::size_t k)
   {
      assert(n > 0 && m > 0 && k > 0);
      auto first = add_variable();
      for(std::size_t i=1; i<n*m*k; ++i) {
         auto l = add_variable();
         assert(l == first+i);
      } 
      return sat_literal_tensor(first,n,m,k);
   }

  template<typename T>
  void add_objective(sat_literal l, const T val)
  {
    assert(l >= 1);
    if(objective_.size() <= l-1) {
      objective_.resize(l, 0.0);
    }
    objective_[l-1] = val;
  }

   template<typename... T_REST>
   void add_clause(T_REST... literals)
   {
      lgladd(sat_,0);
   }
   template<typename T, typename... T_REST>
   void add_clause(T literal, T_REST... literals)
   {
      static_assert(std::is_same<T,sat_literal>::value,"");
      assert(literal != 0 && std::abs(literal) <= size());
      lgladd(sat_, literal);
      add_clause(literals...);
   }

   void fix_literal(const sat_literal l)
   {
      add_clause(l);
   }

   // i => j
   void add_implication(const sat_literal i, const sat_literal j)
   {
      add_clause(-i,j);
   }

   void make_equal(const sat_literal i, const sat_literal j)
   {
      add_implication(i,j);
      add_implication(j,i);
   }

  template<typename ITERATOR>
  sat_literal add_at_most_one_constraint_naive(ITERATOR literal_begin, ITERATOR literal_end)
  {
    //assert(1 < std::distance(literal_begin, literal_end));
    const std::size_t n = std::distance(literal_begin, literal_end);
    if(n == 1) {
      return *literal_begin;
    }
    for(std::size_t i=0; i<n; ++i) {
      for(std::size_t j=i+1; j<n; ++j) {
         add_clause(-*(literal_begin+i), -*(literal_begin+j));
      }
    }
    auto c = add_variable();
    for(std::size_t i=0; i<n; ++i) {
       add_clause(c, -*(literal_begin+i));
    }
    

    lgladd(sat_, -c);
    for(std::size_t i=0; i<n; ++i) {
      lgladd(sat_, *(literal_begin+i));
    }
    lgladd(sat_, 0);

    return c; 
  }

  template<typename ITERATOR>
  void add_simplex_constraint_naive(ITERATOR literal_begin, ITERATOR literal_end)
  {
    const std::size_t n = std::distance(literal_begin, literal_end);
    if(n == 1) {
       add_clause(*literal_begin);
    } else {
       for(std::size_t i=0; i<n; ++i) {
          for(std::size_t j=i+1; j<n; ++j) {
             add_clause( -*(literal_begin+i), -*(literal_begin+j) );
          }
       }

       for(std::size_t i=0; i<n; ++i) {
          lgladd(sat_, *(literal_begin+i));
       }
       lgladd(sat_, 0);
    } 
  }

  template<typename ITERATOR>
  sat_literal add_at_most_one_constraint(ITERATOR literal_begin, ITERATOR literal_end)
  {
    constexpr std::size_t th = 3; // pursue a direct approach until 6 variables, and if more a recursive one
    const std::size_t n = std::distance(literal_begin, literal_end);
    assert(n > 0);
    if(n <= th) {
      return add_at_most_one_constraint_naive(literal_begin, literal_end);
    } else {
      if(n <= 3*th) {
        auto c1 = add_at_most_one_constraint(literal_begin, literal_begin + n/2);
        auto c2 = add_at_most_one_constraint(literal_begin + n/2, literal_end);
        std::array<sat_literal,2> c_list {c1,c2};
        return add_at_most_one_constraint_naive(c_list.begin(), c_list.end()); 
      } else {
        auto c1 = add_at_most_one_constraint(literal_begin, literal_begin + n/3);
        auto c2 = add_at_most_one_constraint(literal_begin + n/3, literal_begin + 2*n/3);
        auto c3 = add_at_most_one_constraint(literal_begin  + 2*n/3, literal_end);
        std::array<sat_literal,3> c_list {c1,c2,c3};
        return add_at_most_one_constraint_naive(c_list.begin(), c_list.end());
      }
    }
  }

  template<typename ITERATOR>
  void add_simplex_constraint(ITERATOR literal_begin, ITERATOR literal_end)
  {
     assert(valid_literal(*literal_begin));
     constexpr std::size_t th = 3; // pursue a direct approach until 6 variables, and if more a recursive one
     const std::size_t n = std::distance(literal_begin, literal_end);
     assert(n > 0);
     if(n <= th) {
        add_simplex_constraint_naive(literal_begin, literal_end);
     } else if(n <= 3*th) {
        auto c1 = add_at_most_one_constraint(literal_begin, literal_begin + n/2);
        auto c2 = add_at_most_one_constraint(literal_begin + n/2, literal_end);
        std::array<sat_literal,2> c_list {c1,c2};
        add_simplex_constraint_naive(c_list.begin(), c_list.end()); 
     } else {
        auto c1 = add_at_most_one_constraint(literal_begin, literal_begin + n/3);
        auto c2 = add_at_most_one_constraint(literal_begin + n/3, literal_begin + 2*n/3);
        auto c3 = add_at_most_one_constraint(literal_begin  + 2*n/3, literal_end);
        std::array<sat_literal,3> c_list {c1,c2,c3};
        add_simplex_constraint_naive(c_list.begin(), c_list.end());
     }
  } 

  template<typename VARIABLE_ITERATOR, typename VALUE_ITERATOR, typename ASSUMPTIONS_VECTOR>
  void reduce_simplex_constraint(VARIABLE_ITERATOR var_begin, VARIABLE_ITERATOR var_end, VALUE_ITERATOR val_begin, VALUE_ITERATOR val_end, ASSUMPTIONS_VECTOR& assumptions, const double th)
  {
    const auto min_val = *std::min_element(val_begin, val_end);
    assert(std::distance(var_begin, var_end) == std::distance(val_begin, val_end));
    auto var_it = var_begin;
    auto val_it = val_begin;
    for(; var_it!=var_end; ++var_it, ++val_it) {
      if(*val_it >= min_val + th) {
        assumptions.push_back({var_begin,0}); 
      } 
    } 
  }

  template<typename ITERATOR>
  sat_literal max(ITERATOR literal_begin, ITERATOR literal_end)
  {
    auto one_active = add_variable();
    for(auto it=literal_begin; it!=literal_end; ++it) {
       add_implication(*it, one_active);
    }

    for(auto it=literal_begin; it!=literal_end; ++it) {
       add_implication(-one_active, -*it);
    }

    // add implication one_active => exists active literal
    lgladd(sat_, -one_active);
    for(auto it=literal_begin; it!=literal_end; ++it) {
       lgladd(sat_, *it);
    }
    lgladd(sat_,0);

    return one_active;
  }

  template<typename ITERATOR>
  sat_literal min(ITERATOR literal_begin, ITERATOR literal_end)
  {
    auto min_literal = add_variable();
    for(auto it=literal_begin; it!=literal_end; ++it) {
       add_implication(-*it, -min_literal);
    }

    for(auto it=literal_begin; it!=literal_end; ++it) {
       add_implication(min_literal, *it);
    } 
    return min_literal;
  }

  // sort networks
  template<typename ITERATOR>
  static void riffle(ITERATOR literal_begin, ITERATOR literal_end)
  {
     sat_vec tmp(literal_begin, literal_end);
     const std::size_t n = std::distance(literal_begin, literal_end);
     for(std::size_t i=0; i<n/2; i++){
        *(literal_begin+i*2) = tmp[i];
        *(literal_begin+i*2+1) = tmp[i + (n/2)];
     }
  }

  template<typename ITERATOR>
  static void unriffle(ITERATOR literal_begin, ITERATOR literal_end)
  {
     sat_vec tmp(literal_begin, literal_end);
     const std::size_t n = std::distance(literal_begin, literal_end);
     for(std::size_t i=0; i<n/2; i++){
        *literal_begin = tmp[i*2];
        *literal_begin + i + (n/2) = tmp[i*2+1];
     }
  }

  template<typename ITERATOR>
  void odd_even_merge(ITERATOR literal_begin, ITERATOR literal_end)
  {
     const std::size_t n = std::distance(literal_begin, literal_end);
     assert(n > 1);
     if (n == 2) {
        auto min = this->min(*literal_begin, *(literal_begin+1));
        auto max = this->max(*literal_begin, *(literal_begin+1));
        *literal_begin = min;
        *(literal_begin+1) = max;
     } else {
        int mid = n / 2;
        sat_vec tmp(literal_begin, literal_end);
        unriffle(tmp.begin(), tmp.end());
        odd_even_merge(tmp.begin(), tmp.begin() + mid);
        odd_even_merge(tmp.begin()+mid, tmp.end());
        riffle(tmp.begin(), tmp.end());
        for(auto it=tmp.begin()+1; it!=tmp.end()-1; it+=2) {
           auto min = this->min(*it, *(it+1));
           auto max = this->max(*it, *(it+1));
           *it = min;
           *(it+1) = max;
        }
        assert(false); // copt tmp back to literal_begin
        std::copy(literal_begin, literal_end, tmp.begin());
     }
  }

  // sort input literals in ascending order
  template<typename ITERATOR>
  sat_vec odd_even_sort(ITERATOR literal_begin, ITERATOR literal_end)
  {
    std::size_t orig_sz = std::distance(literal_begin, literal_end);
     assert(orig_sz >= 2);
     std::size_t sz; for (sz = 1; sz < orig_sz; sz *= 2);
     sat_vec literals(sz);
     auto it = literal_begin;
     auto it_copy = literals.begin();
     for(; it!=literal_end; ++it, ++it_copy) {
        *it_copy = *it;
     }
     for(; it_copy!=literals.end(); ++it_copy) {
        assert(false); // pad with constant 1
        *it_copy = *literal_begin;
     }

     for (int i = 1; i < literals.size(); i *= 2) {
        for (int j = 0; j + 2*i <= literals.size(); j += 2*i) {
           odd_even_merge(literals.begin()+j, literals.begin()+j+2*i);
        }
     }

     sat_vec ret_vec(literals.begin(), literals.begin()+orig_sz);
     return std::move(ret_vec); 
  }

  // solution functions
  bool solve()
  {
     for(std::size_t i=0; i<size(); ++i) {
        lglfreeze(sat_, to_literal(i));
     }
     const int sat_ret = lglsat(sat_);

     const bool feasible = (sat_ret == LGL_SATISFIABLE);
     return feasible; 
  }

  bool solution(const sat_literal l) const
  {
     assert(l != 0);
     assert(lglderef(sat_, l) != 0);
     return lglderef(sat_, l) == 1; 
  }

  template<typename ITERATOR>
  bool solve(ITERATOR begin, ITERATOR end)
  {
     for(auto it=begin; it!=end; ++it) {
        assert(valid_literal(*it));
        lglassume( sat_, *it );
     }
     for(std::size_t i=0; i<size(); ++i) {
        lglfreeze(sat_, to_literal(i));
     }
     return solve();
  }

  std::vector<bool> solution()
  {
     std::vector<bool> solution;
     solution.reserve(size());
     for(std::size_t i=0; i<size(); ++i) {
        solution.push_back(lglderef(sat_, i+1) == 1);
     }
     return std::move(solution);
  }

  /*
  void assume_literal(sat_literal l)
  {
     assert(false);
     lglassume(sat_, l);
  }
  */

protected:
  
  LGL* const sat_;
  std::vector<double> objective_;
};

/*
class sat_opt_solver : public sat_solver {
public:
   using sat_solver::sat_solver;
   sat_opt_solver() : sat_solver::solver()
   {
      picosat_enable_trace_generation(this->s_);
   }

   void solve()
   {
      assert(costs_.size == this->size());
      // first of all, assume that problem is feasible
      
      //res = picosat_sat (ps, -1);
      set_assumptions();

      while(auto status = picosat_sat(this->sat_) != PICOSAT_SATISFIABLE) {
         assert(status == PICOSAT_UNSATISFIABLE);

         // extract unsatisfiable core
         auto q = picosat_mus_assumptions (this->s_, 0, callback, 1);
         std::vector<sat_literal> unsat_core;
         while(*q) { 
            unsat_core.push_back(*q); 
         }

         // get minimal cost among literals in unsatisfiable core, increase lb by that value
         min_cost = std::numeric_limits<REAL>::infinity();
         for(sat_literal l : unsat_core) {
            const std::size_t var = std::abs(l) - 1;
            min_cost = std::min( std::abs(costs_[var]), min_cost );
         }

         // adjust costs: make absolute value of costs of literals in unsat core smaller by min_val, set costs of sorted network elements to min_val
         for(sat_literal l : unsat_core) {
            const std::size_t var = std::abs(l) - 1;
            costs_[var] -= costs_[var] > 0 ? min_cost : -min_cost;
         }
         
         // given unsatisfiable literals, construct sorting network on those.
         auto sorted = this->odd_even_sort(unsat_core.begin(), unsat_core.end());

         // at least one element must change in unsat core. Hence we can add constraint that first sorted element is involuted.
         this->fix_variable(-sorted[0]); assert(false); // or last one for ascending order?

         // extend cost vector by new elements
         costs_.resize(this->size(), 0.0);
         for(const auto literal : sorted) {
            costs_[literal] = min_cost;
         } 

         set_assumptions(); 
      }
   }

   void set_cost(sat_literal l, const REAL cost) 
   { 
      assert(false); // unfix literal
      cost_[to_var(l)] = cost;
   }

   // to do: allow for threshold
   void set_assumptions()
   {
      for(std::size_t i=0; i<cost_.size(); ++i) {
         if(cost_[i] < 0.0) {
            picosat_assume( this->s_, this->to_literal(i) );
         } else if(cost_[i] > 0.0) {
            picosat_assume( this->s_, -this->to_literal(i) );
         }
      }
      //for(std::size_t i=0; i<this->size(); ++i) {
      //   lglfreeze(this->s_, to_literal(i));
      //}
   }

private:
   std::vector<REAL> costs_;
};
*/

} // end namespace LP_MP

#endif // DD_ILP_SAT_SOLVER_HXX
