#ifndef PROBLEM_EXPORT_HXX
#define PROBLEM_EXPORT_HXX

#include <iostream>
#include <vector>
#include <fstream>
#include <iterator>

namespace DD_ILP {

// store problem and print it out to an .lp file
class problem_export {
public:

struct vector_iterator : std::iterator< std::random_access_iterator_tag, std::size_t >{
   vector_iterator(std::size_t _l) : l(_l) {}
   bool operator==(const vector_iterator o) const { return l == o.l; }
   bool operator!=(const vector_iterator o) const { return !(*this == o); }
   void operator++() { ++l; }
   const std::size_t operator*() const { return l; }
   long operator-(vector_iterator o) const { return l - o.l; }
   vector_iterator operator+(const int n) const { return vector_iterator(l + n); }
   vector_iterator operator-(const int n) const { return vector_iterator(l - n); }
   private:
   std::size_t l;
};

struct vector_iterator_strided : std::iterator< std::random_access_iterator_tag, std::size_t >{
   vector_iterator_strided(std::size_t _l, const std::size_t _stride) : l(_l), stride(_stride) {}
   bool operator==(const vector_iterator_strided o) const { return (l == o.l && stride == o.stride); }
   bool operator!=(const vector_iterator_strided o) const { return !(*this == o); }
   void operator++() { l+=stride; }
   const std::size_t operator*() const { return l; }
   long operator-(vector_iterator_strided o) const { return (l - o.l)/stride; }
   vector_iterator_strided operator+(const int n) const { return vector_iterator_strided(l + n*stride, stride); } 
   private:
   std::size_t l;
   std::size_t stride;
};


struct vector {
   template<typename VECTOR>
   vector(const VECTOR& v) : begin_(0), dim_(v.size()) {}

   vector(const std::size_t dim) : begin_(0), dim_(dim) 
   {
      assert(dim > 0);
   } 
   vector(std::size_t f, const std::size_t dim) : begin_(f), dim_(dim) 
   {
      assert(dim > 0);
   } 
   void set_begin(std::size_t l)
   {
      begin_ = l;
   }
   bool operator==(const vector o) const
   {
      return begin_ == o.begin_ && size() == o.size();
   }

   std::size_t size() const { return dim_; }
   const std::size_t operator[](const std::size_t i) const { assert(i < size()); return begin_ + i; }
   const std::size_t back() const { return begin_ + size()-1; }

   auto begin() const {
      return vector_iterator(begin_);
   }
   auto end() const {
      return vector_iterator(begin_ + size());
   }
   private:
   std::size_t begin_;
   const std::size_t dim_;
};

struct vector_strided {
   vector_strided(std::size_t f, const std::size_t dim, const std::size_t stride) : begin_(f), dim_(dim), stride_(stride)
   {
      assert(dim > 0 && stride_ > 0);
   } 
   bool operator==(const vector_strided o) const
   {
      return begin_ == o.begin_ && size() == o.size() && stride_ == o.stride_;
   }

   std::size_t size() const { return dim_; }
   const std::size_t operator[](const std::size_t i) const { assert(i < size()); return begin_ + i*stride_; }
   const std::size_t back() const { return (*this)[size()-1]; }

   auto begin() const {
      return vector_iterator_strided(begin_, stride_);
   }
   auto end() const {
      return vector_iterator_strided(begin_ + stride_*size(), stride_);
   }
   private:
   std::size_t begin_;
   const std::size_t dim_;
   const std::size_t stride_;
}; 

struct matrix {
   template<typename MATRIX>
   matrix(const MATRIX& m) : begin_(0), dim_({m.dim1(), m.dim2()}) {}

   matrix(const std::size_t n, const std::size_t m) : begin_(0), dim_({n,m}) 
   {
      assert(n > 0 && m > 0); 
   }
   matrix(std::size_t f, const std::size_t n, const std::size_t m) : begin_(f), dim_({n,m}) 
   {
      assert(n > 0 && m > 0);
   } 
   void set_begin(std::size_t l)
   {
      begin_ = l;
   }
   bool operator==(const matrix& o) const
   {
      return begin_ == o.begin_ && dim_ == o.dim_;
   }

   std::size_t size() const { return dim_[0]*dim_[1]; }
   std::size_t dim1() const { return dim_[0]; }
   std::size_t dim2() const { return dim_[1]; }
   std::size_t dim(const std::size_t d) const { assert(d<2); return dim_[d]; }
   const std::size_t operator[](const std::size_t i) const 
   {
      assert(i < size());
      return begin_ + i;
   }
   const std::size_t operator()(const std::size_t x1, const std::size_t x2) const 
   { 
      assert(x1 < dim(0) && x2 < dim(1));
      return begin_ + x1*dim(1) + x2;
   }

   auto begin() const {
      return vector_iterator(begin_);
   }
   auto end() const {
      return vector_iterator(begin_ + size());
   }

   vector slice_left(const std::size_t x1) const {
      assert(x1 < dim(0));
      return vector(begin_ + x1*dim(1),dim(1));
   }
   auto slice1(const std::size_t x1) const { return slice_left(x1); }

   vector_strided slice_right(const std::size_t x2) const {
      assert(x2 < dim(1));
      return vector_strided(begin_ + x2, dim(0), dim(1));
   }
   auto slice2(const std::size_t x2) const { return slice_right(x2); }
   private:
   std::size_t begin_;
   const std::array<std::size_t,2> dim_;
};

struct tensor {
   template<typename TENSOR>
   tensor(const TENSOR& t) : begin_(0), dim_({t.dim1(), t.dim2(), t.dim3()}) {}

   tensor(const std::size_t n, const std::size_t m, const std::size_t k) : begin_(0), dim_({n,m,k}) 
   {
      assert(n > 0 && m > 0 && k > 0);
   } 
   tensor(std::size_t f, const std::size_t n, const std::size_t m, const std::size_t k) : begin_(f), dim_({n,m,k}) 
   {
      assert(n > 0 && m > 0 && k > 0);
   } 
   void set_begin(std::size_t l)
   {
      assert(l > 0);
      begin_ = l;
   }
   bool operator==(const tensor& o) const
   {
      return begin_ == o.begin_ && dim_ == o.dim_;
   }

   std::size_t size() const { return dim_[0]*dim_[1]*dim_[2]; }
   std::size_t dim(const std::size_t d) const { assert(d<3); return dim_[d]; }
   const std::size_t operator[](const std::size_t i) const 
   {
      assert(i < size());
      return begin_ + i;
   }
   const std::size_t operator()(const std::size_t x1, const std::size_t x2, const std::size_t x3) const 
   { 
      assert(x1 < dim(0) && x2 < dim(1) && x3 < dim(2));
      return begin_ + x1*dim(1)*dim(2) + x2*dim(2) + x3;
   }

   auto begin() const {
      return vector_iterator(begin_);
   }
   auto end() const {
      return vector_iterator(begin_ + size());
   }

   vector slice12(const std::size_t x1, const std::size_t x2) const
   {
      assert(x1 < dim(0) && x2 < dim(1));
      return vector(begin_ + x1*dim(2)*dim(1) + x2*dim(2),dim(2)); 
   }
   vector_strided slice13(const std::size_t x1, const std::size_t x3) const
   {
      assert(x1 < dim(0) && x3 < dim(2));
      return vector_strided(begin_ + x1*dim(2)*dim(1) + x3, dim(1), dim(2)); 
   }
   vector_strided slice23(const std::size_t x2, const std::size_t x3) const
   {
      assert(x2 < dim(1) && x3 < dim(2));
      return vector_strided(begin_ + x2*dim(2) + x3, dim(0), dim(1)*dim(2)); 
   }

   private:
   std::size_t begin_;
   const std::array<std::size_t,3> dim_;
};
  
  using variable = std::size_t;

  variable add_variable() { return variable_counter_++; }
  vector add_vector(const std::size_t dim)
  {
    vector vec(variable_counter_, dim);
    variable_counter_ += dim;
    return vec;
  }
  matrix add_matrix(const std::size_t n, const std::size_t m)
  {
    matrix mat(variable_counter_, n, m);
    variable_counter_ += n*m;
    return mat;
  }
  tensor add_tensor(const std::size_t n, const std::size_t m, const std::size_t k)
  {
    tensor t(variable_counter_, n, m, k);
    variable_counter_ += n*m*k;
    return t;
  }
  template<typename T>
  void add_objective(variable x, const T val)
  {
    if(objective_.size() <= x) {
      objective_.resize(x+1, 0.0);
    }
    objective_[x] = val;
  }
  void add_implication(const variable i, const variable j)
  {
    constraints_.push_back(var_name(i) + " <= " + var_name(j));
  }
  template<typename ITERATOR>
  variable add_at_most_one_constraint(ITERATOR variable_begin, ITERATOR variable_end)
  {
    auto sum_var = add_variable();
    std::string c = sum(variable_begin, variable_end);
    c += " - " + var_name(sum_var) + " = 0";
    constraints_.push_back(std::move(c));
    return sum_var;
  }
  template<typename ITERATOR>
  void add_simplex_constraint(ITERATOR variable_begin, ITERATOR variable_end)
  {
    auto c = sum(variable_begin, variable_end) += " = 1";
    constraints_.push_back(std::move(c));
  }
  void make_equal(const variable i, const variable j)
  {
    constraints_.push_back(var_name(i) + " = " + var_name(j));
  }
  template<typename ITERATOR>
  variable max(ITERATOR var_begin, ITERATOR var_end)
  {
    auto one_active = add_variable();
    for(auto it=var_begin; it!=var_end; ++it) {
       add_implication(*it, one_active);
    }

    for(auto it=var_begin; it!=var_end; ++it) {
       constraints_.push_back("1 - " + var_name(one_active) + " >= " + var_name(*it));
    }

    // add implication one_active => exists active variable
    constraints_.push_back(var_name(one_active) + " <= ");
    constraints_.back() += sum(var_begin, var_end);

    return one_active;
  } 

  void write_to_file(const std::string& filename) const
  {
    std::ofstream f(filename);
    if(!f) {
      throw std::runtime_error("could not open file" + filename + " for LP-export");
    }

    f << "Minimize\n";
    for(variable i=0; i<objective_.size(); ++i) {
      auto obj = objective_[i];
      f << (obj < 0 ? std::to_string(obj) : "+" + std::to_string(obj)) << " " << var_name(i) << "\n";
    }

    f << "Subject To\n";
    for(auto& c : constraints_) {
      f << c << "\n";
    }

    f << "Bounds\nBinaries\n";
    for(variable i=0; i<variable_counter_; ++i) {
      f << var_name(i) << "\n";
    }

    f << "End";

    f.close();
  }

private:
  static std::string var_name(const variable x) 
  {
    return "x" + std::to_string(x); 
  }
  template<typename ITERATOR>
  static std::string sum(ITERATOR var_begin, ITERATOR var_end)
  {
    assert(std::distance(var_begin, var_end) > 0);
    std::string c = var_name(*var_begin);
    for(auto it=var_begin+1; it!=var_end; ++it) {
      c += " + " + var_name(*it);
    }
    return c; 
  }
  std::size_t variable_counter_ = 0;
  std::vector<double> objective_;
  std::vector<std::string> constraints_;

};

} // namespace DD_ILP

#endif // PROBLEM_EXPORT_HXX

