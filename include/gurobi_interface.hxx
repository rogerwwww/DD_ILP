#ifndef DD_ILP_GUROBI_INTERFACE_HXX
#define DD_ILP_GUROBI_INTERFACE_HXX

#include "gurobi_c++.h"

namespace DD_ILP {

class gurobi_interface {
public:
  gurobi_interface()
    : model_(env_)
  {}
  using variable = GRBVar;

  struct gurobi_variable_vector_iterator_strided {
    gurobi_variable_vector_iterator_strided(GRBVar* it, const std::size_t stride) : it_(it), stride_(stride) {}
    bool operator==(const gurobi_variable_vector_iterator_strided o) const { return (it_ == o.it_ && stride_ == o.stride_); }
    bool operator!=(const gurobi_variable_vector_iterator_strided o) const { return !(*this == o); }
    void operator++() { it_+=stride_; }
    const GRBVar operator*() const { return *it_; }
    int operator-(gurobi_variable_vector_iterator_strided o) const { return (it_ - o.it_)/stride_; }
    gurobi_variable_vector_iterator_strided operator+(const int n) const { return gurobi_variable_vector_iterator_strided(it_ + n*stride_, stride_); } 
    private:
    GRBVar* it_;
    std::size_t stride_;
  };

  struct gurobi_variable_vector {
    gurobi_variable_vector(const std::size_t dim) : begin_(0), dim_(dim) 
    {
      assert(dim > 0);
    } 
    gurobi_variable_vector(GRBVar* v, const std::size_t dim) : begin_(v), dim_(dim) 
    {
      assert(dim > 0);
    } 
    void set_begin(GRBVar* v)
    {
      assert(v > 0);
      begin_ = v;
    }
    bool operator==(const gurobi_variable_vector o) const
    {
      return begin_ == o.begin_ && size() == o.size();
    }

    std::size_t size() const { return dim_; }
    const GRBVar& operator[](const std::size_t i) const { assert(i < size()); return *(begin_ + i); }
    const GRBVar& back() const { return *(begin_ + size()-1); }

    auto begin() const {
      return begin_;
    }
    auto end() const {
      return begin_ + size();
    }
    private:
    GRBVar* begin_;
    const std::size_t dim_;
  };

  struct gurobi_variable_vector_strided {
    gurobi_variable_vector_strided(GRBVar* begin, const std::size_t dim, const std::size_t stride) : begin_(begin), dim_(dim), stride_(stride)
    {
      assert(dim > 0 && stride_ > 0);
      assert(begin > 0);
    } 
    bool operator==(const gurobi_variable_vector_strided o) const
    {
      return begin_ == o.begin_ && size() == o.size() && stride_ == o.stride_;
    }

    std::size_t size() const { return dim_; }
    const GRBVar operator[](const std::size_t i) const { assert(i < size()); return *(begin_ + i*stride_); }
    const GRBVar back() const { return (*this)[size()-1]; }

    auto begin() const {
      return gurobi_variable_vector_iterator_strided(begin_, stride_);
    }
    auto end() const {
      return gurobi_variable_vector_iterator_strided(begin_ + stride_*size(), stride_);
    }
    private:
    GRBVar* begin_;
    const std::size_t dim_;
    const std::size_t stride_;
  }; 

  struct gurobi_variable_matrix {
   gurobi_variable_matrix(const std::size_t n, const std::size_t m) : begin_(0), dim_({n,m}) 
   {
      assert(n > 0 && m > 0); 
   }
   gurobi_variable_matrix(GRBVar* begin, const std::size_t n, const std::size_t m) : begin_(begin), dim_({n,m}) 
   {
      assert(n > 0 && m > 0);
      assert(begin > 0);
   } 
   void set_begin(GRBVar* begin)
   {
      assert(begin > 0);
      begin_ = begin;
   }
   bool operator==(const gurobi_variable_matrix& o) const
   {
      return begin_ == o.begin_ && dim_ == o.dim_;
   }

   std::size_t size() const { return dim_[0]*dim_[1]; }
   std::size_t dim1() const { return dim_[0]; }
   std::size_t dim2() const { return dim_[1]; }
   std::size_t dim(const std::size_t d) const { assert(d<2); return dim_[d]; }
   const GRBVar& operator[](const std::size_t i) const 
   {
      assert(begin_ > 0);
      assert(i < size());
      return *(begin_ + i);
   }
   const GRBVar& operator()(const std::size_t x1, const std::size_t x2) const 
   { 
      assert(begin_ > 0);
      assert(x1 < dim(0) && x2 < dim(1));
      return *(begin_ + x1*dim(1) + x2);
   }

   auto begin() const {
      return begin_;
   }
   auto end() const {
      return begin_ + size();
   }

   gurobi_variable_vector slice_left(const std::size_t x1) const {
      assert(x1 < dim(0));
      return gurobi_variable_vector(begin_ + x1*dim(1),dim(1));
   }
   auto slice1(const std::size_t x1) const { return slice_left(x1); }

   gurobi_variable_vector_strided slice_right(const std::size_t x2) const {
      assert(x2 < dim(1));
      return gurobi_variable_vector_strided(begin_ + x2, dim(0), dim(1));
   }
   auto slice2(const std::size_t x2) const { return slice_right(x2); }
   private:
   GRBVar* begin_;
   const std::array<std::size_t,2> dim_;
  };

  struct gurobi_variable_tensor {
    gurobi_variable_tensor(const std::size_t n, const std::size_t m, const std::size_t k) : begin_(0), dim_({n,m,k}) 
    {
      assert(n > 0 && m > 0 && k > 0);
    } 
    gurobi_variable_tensor(GRBVar* begin, const std::size_t n, const std::size_t m, const std::size_t k) : begin_(begin), dim_({n,m,k}) 
    {
      assert(n > 0 && m > 0 && k > 0);
      assert(begin > 0);
    } 
    void set_begin(GRBVar* begin)
    {
      assert(begin > 0);
      begin_ = begin;
    }
    bool operator==(const gurobi_variable_tensor& o) const
    {
      return begin_ == o.begin_ && dim_ == o.dim_;
    }

    std::size_t size() const { return dim_[0]*dim_[1]*dim_[2]; }
    std::size_t dim(const std::size_t d) const { assert(d<3); return dim_[d]; }
    const GRBVar& operator[](const std::size_t i) const 
    {
      assert(i < size());
      return *(begin_ + i);
    }
    const GRBVar& operator()(const std::size_t x1, const std::size_t x2, const std::size_t x3) const 
    { 
      assert(x1 < dim(0) && x2 < dim(1) && x3 < dim(2));
      return *(begin_ + x1*dim(1)*dim(2) + x2*dim(2) + x3);
    }

    auto begin() const {
      return begin_;
    }
    auto end() const {
      return begin_ + size();
    }

    gurobi_variable_vector slice12(const std::size_t x1, const std::size_t x2) const
    {
      assert(x1 < dim(0) && x2 < dim(1));
      return gurobi_variable_vector(begin_ + x1*dim(2)*dim(1) + x2*dim(2),dim(2)); 
    }
    gurobi_variable_vector_strided slice13(const std::size_t x1, const std::size_t x3) const
    {
      assert(x1 < dim(0) && x3 < dim(2));
      return gurobi_variable_vector_strided(begin_ + x1*dim(2)*dim(1) + x3, dim(1), dim(2)); 
    }
    gurobi_variable_vector_strided slice23(const std::size_t x2, const std::size_t x3) const
    {
      assert(x2 < dim(1) && x3 < dim(2));
      return gurobi_variable_vector_strided(begin_ + x2*dim(2) + x3, dim(0), dim(1)*dim(2)); 
    }

    private:
    GRBVar* begin_;
    const std::array<std::size_t,3> dim_;
  };


  using vector = gurobi_variable_vector;
  using matrix = gurobi_variable_matrix;
  using tensor = gurobi_variable_tensor;

  GRBVar add_variable()
  {
    return model_.addVar(0.0,1.0,0.0,GRB_BINARY);
  }

  gurobi_variable_vector add_vector(const std::size_t dim)
  {
    GRBVar* vars = model_.addVars(dim, GRB_BINARY);
    return gurobi_variable_vector(vars, dim);
  }
  gurobi_variable_matrix add_matrix(const std::size_t n, const std::size_t m)
  {
    GRBVar* vars = model_.addVars(n*m, GRB_BINARY);
    return gurobi_variable_matrix(vars, n, m);
  }
  gurobi_variable_tensor add_tensor(const std::size_t n, const std::size_t m, const std::size_t l)
  {
    GRBVar* vars = model_.addVars(n*m*l, GRB_BINARY);
    return gurobi_variable_tensor(vars, n, m, l);
  }

  template<typename T>
  void add_objective(const GRBVar& x, const T val)
  {
    // possibly remove previous instance of term?
    const double _val(val);
    objective_.remove(x);
    objective_.addTerms(&_val, &x, 1);
  }
  //////////////////////////////////////
  // functions for adding constraints //
  //////////////////////////////////////
  
   void add_implication(const GRBVar& i, const GRBVar& j)
   {
     model_.addConstr(i <= j);
   }
   template<typename ITERATOR>
   GRBVar max(ITERATOR begin, ITERATOR end)
   {
     auto max_var = model_.addVar(0.0, 1.0, 0.0, GRB_BINARY);
     for(auto it=begin; it!=end; ++it) {
       add_implication(*it, max_var);
     }
     GRBLinExpr sum;
     for(auto it=begin; it!=end; ++it) {
       sum += *it;
     }
     model_.addConstr(sum >= max_var);

     return max_var; 
   }

   template<typename ITERATOR>
   GRBVar add_at_most_one_constraint(ITERATOR variable_begin, ITERATOR variable_end)
   {
     GRBLinExpr sum_expr = 0;
     auto sum = model_.addVar(0.0, 1.0, 0.0, GRB_BINARY);
     for(auto it=variable_begin; it!=variable_end; ++it) {
       sum_expr += *it;
     }
     sum_expr -= sum;
     model_.addConstr(sum_expr == 0);
     return sum;
   }

  template<typename ITERATOR>
  void add_simplex_constraint(ITERATOR variable_begin, ITERATOR variable_end)
  {
     GRBLinExpr sum_expr = 0;
     for(auto it=variable_begin; it!=variable_end; ++it) {
       sum_expr += *it;
     }
     model_.addConstr(sum_expr == 1);
  }

   void make_equal(const GRBVar& i, const GRBVar& j)
   {
     model_.addConstr(i == j);
   }

   ////////////////////////
   // solution functions //
   ////////////////////////

   bool solve()
   {
     model_.update();
     model_.setObjective(objective_, GRB_MINIMIZE);
     model_.optimize();
     return model_.get(GRB_IntAttr_Status) == GRB_OPTIMAL;
   }

   bool solution(const GRBVar& var) const
   {
     assert(var.get(GRB_DoubleAttr_X) < 0.01 || var.get(GRB_DoubleAttr_X) > -0.01 || var.get(GRB_DoubleAttr_X) > 0.99 || var.get(GRB_DoubleAttr_X) < 1.01);
     return var.get(GRB_DoubleAttr_X) > 0.99;
   }

   void write_to_file(const std::string& filename)
   {
     model_.update();
     model_.setObjective(objective_, GRB_MINIMIZE);
     model_.write(filename);
   }

private:
  GRBEnv env_;
  GRBModel model_;
  GRBLinExpr objective_;
};

} // end namespace DD_ILP

#endif // DD_ILP_GUROBI_INTERFACE_HXX
