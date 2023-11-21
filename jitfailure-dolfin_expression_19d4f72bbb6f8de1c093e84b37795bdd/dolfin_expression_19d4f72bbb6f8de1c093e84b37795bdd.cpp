
// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

#include <dolfin/function/Expression.h>
#include <dolfin/math/basic.h>
#include <Eigen/Dense>


// cmath functions
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cosh;
using std::sinh;
using std::tanh;
using std::exp;
using std::frexp;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::sqrt;
using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;
using std::max;
using std::min;

const double pi = DOLFIN_PI;


namespace dolfin
{
  class dolfin_expression_19d4f72bbb6f8de1c093e84b37795bdd : public Expression
  {
     public:
       double t;


       dolfin_expression_19d4f72bbb6f8de1c093e84b37795bdd()
       {
            _value_shape.push_back(2);
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          values[0] = t*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(3.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**2*(x[1] - 1)**3 + 3.0*t**3*x[0]**2*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**3 - 1.5*t**2*x[0]**2*x[1]**2*(2*x[0] - 2)*(x[1] - 1)**2 - 3.0*t**2*x[0]*x[1]**2*(x[0] - 1)**2*(x[1] - 1)**2) + t*x[0]*x[1]*(x[1] - 1)*(1.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**3 - 1.5*t**2*x[0]**2*x[1]**2*(x[0] - 1)**2*(x[1] - 1)**2 + 1) + t*x[1]*(x[0] - 1)*(x[1] - 1)*(1.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**3 - 1.5*t**2*x[0]**2*x[1]**2*(x[0] - 1)**2*(x[1] - 1)**2 + 1) - (-0.3*t*x[0]*x[1]*(x[1] - 1) - 0.3*t*x[1]*(x[0] - 1)*(x[1] - 1) + 2*t*x[1]*(x[1] - 1))*(6.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**3 - 9.0*t**2*x[0]**2*x[1]**2*(x[0] - 1)**2*(x[1] - 1)**2 + 4.0) - (-0.3*t*x[0]*x[1]*(x[0] - 1)*(x[1] - 1) + t*x[0]*x[1]*(x[0] - 1) + t*x[0]*(x[0] - 1)*(x[1] - 1))*(9.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**2*(x[1] - 1)**3 + 9.0*t**3*x[0]**2*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**3 - 4.5*t**2*x[0]**2*x[1]**2*(2*x[0] - 2)*(x[1] - 1)**2 - 9.0*t**2*x[0]*x[1]**2*(x[0] - 1)**2*(x[1] - 1)**2) - (-0.3*t*x[0]*x[1]*(x[0] - 1)*(x[1] - 1) + t*x[0]*x[1]*(x[1] - 1) + t*x[1]*(x[0] - 1)*(x[1] - 1))*(18.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**2*(x[1] - 1)**3 + 18.0*t**3*x[0]**2*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**3 - 9.0*t**2*x[0]**2*x[1]**2*(2*x[0] - 2)*(x[1] - 1)**2 - 18.0*t**2*x[0]*x[1]**2*(x[0] - 1)**2*(x[1] - 1)**2) - (3.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**3 - 4.5*t**2*x[0]**2*x[1]**2*(x[0] - 1)**2*(x[1] - 1)**2 + 2.0)*(-0.3*t*x[0]*x[1]*(x[1] - 1) + t*x[0]*x[1] + t*x[0]*(x[1] - 1) - 0.3*t*x[1]*(x[0] - 1)*(x[1] - 1) + t*x[1]*(x[0] - 1) + t*(x[0] - 1)*(x[1] - 1)) - (12.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**3 - 18.0*t**2*x[0]**2*x[1]**2*(x[0] - 1)**2*(x[1] - 1)**2 + 8.0)*(0.5*t*x[0]*x[1] + 1.0*t*x[0]*(x[0] - 1) + 0.5*t*x[0]*(x[1] - 1) + 0.5*t*x[1]*(x[0] - 1) + 0.5*t*(x[0] - 1)*(x[1] - 1)) - (0.5*t*x[0]*x[1]*(x[0] - 1) + 0.5*t*x[0]*x[1]*(x[1] - 1) + 0.5*t*x[0]*(x[0] - 1)*(x[1] - 1) + 0.5*t*x[1]*(x[0] - 1)*(x[1] - 1))*(36.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**2 + 36.0*t**3*x[0]**3*x[1]**2*(x[0] - 1)**3*(x[1] - 1)**3 - 18.0*t**2*x[0]**2*x[1]**2*(x[0] - 1)**2*(2*x[1] - 2) - 36.0*t**2*x[0]**2*x[1]*(x[0] - 1)**2*(x[1] - 1)**2);
          values[1] = t*x[0]*x[1]*(x[0] - 1)*(x[1] - 1)*(3.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**2 + 3.0*t**3*x[0]**3*x[1]**2*(x[0] - 1)**3*(x[1] - 1)**3 - 1.5*t**2*x[0]**2*x[1]**2*(x[0] - 1)**2*(2*x[1] - 2) - 3.0*t**2*x[0]**2*x[1]*(x[0] - 1)**2*(x[1] - 1)**2) + t*x[0]*x[1]*(x[0] - 1)*(1.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**3 - 1.5*t**2*x[0]**2*x[1]**2*(x[0] - 1)**2*(x[1] - 1)**2 + 1) + t*x[0]*(x[0] - 1)*(x[1] - 1)*(1.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**3 - 1.5*t**2*x[0]**2*x[1]**2*(x[0] - 1)**2*(x[1] - 1)**2 + 1) - (-0.3*t*x[0]*x[1]*(x[0] - 1) - 0.3*t*x[0]*(x[0] - 1)*(x[1] - 1) + 2*t*x[0]*(x[0] - 1))*(6.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**3 - 9.0*t**2*x[0]**2*x[1]**2*(x[0] - 1)**2*(x[1] - 1)**2 + 4.0) - (-0.3*t*x[0]*x[1]*(x[0] - 1)*(x[1] - 1) + t*x[0]*x[1]*(x[0] - 1) + t*x[0]*(x[0] - 1)*(x[1] - 1))*(18.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**2 + 18.0*t**3*x[0]**3*x[1]**2*(x[0] - 1)**3*(x[1] - 1)**3 - 9.0*t**2*x[0]**2*x[1]**2*(x[0] - 1)**2*(2*x[1] - 2) - 18.0*t**2*x[0]**2*x[1]*(x[0] - 1)**2*(x[1] - 1)**2) - (-0.3*t*x[0]*x[1]*(x[0] - 1)*(x[1] - 1) + t*x[0]*x[1]*(x[1] - 1) + t*x[1]*(x[0] - 1)*(x[1] - 1))*(9.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**2 + 9.0*t**3*x[0]**3*x[1]**2*(x[0] - 1)**3*(x[1] - 1)**3 - 4.5*t**2*x[0]**2*x[1]**2*(x[0] - 1)**2*(2*x[1] - 2) - 9.0*t**2*x[0]**2*x[1]*(x[0] - 1)**2*(x[1] - 1)**2) - (3.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**3 - 4.5*t**2*x[0]**2*x[1]**2*(x[0] - 1)**2*(x[1] - 1)**2 + 2.0)*(-0.3*t*x[0]*x[1]*(x[0] - 1) + t*x[0]*x[1] - 0.3*t*x[0]*(x[0] - 1)*(x[1] - 1) + t*x[0]*(x[1] - 1) + t*x[1]*(x[0] - 1) + t*(x[0] - 1)*(x[1] - 1)) - (12.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**3 - 18.0*t**2*x[0]**2*x[1]**2*(x[0] - 1)**2*(x[1] - 1)**2 + 8.0)*(0.5*t*x[0]*x[1] + 0.5*t*x[0]*(x[1] - 1) + 0.5*t*x[1]*(x[0] - 1) + 1.0*t*x[1]*(x[1] - 1) + 0.5*t*(x[0] - 1)*(x[1] - 1)) - (0.5*t*x[0]*x[1]*(x[0] - 1) + 0.5*t*x[0]*x[1]*(x[1] - 1) + 0.5*t*x[0]*(x[0] - 1)*(x[1] - 1) + 0.5*t*x[1]*(x[0] - 1)*(x[1] - 1))*(36.0*t**3*x[0]**3*x[1]**3*(x[0] - 1)**2*(x[1] - 1)**3 + 36.0*t**3*x[0]**2*x[1]**3*(x[0] - 1)**3*(x[1] - 1)**3 - 18.0*t**2*x[0]**2*x[1]**2*(2*x[0] - 2)*(x[1] - 1)**2 - 36.0*t**2*x[0]*x[1]**2*(x[0] - 1)**2*(x[1] - 1)**2);

       }

       void set_property(std::string name, double _value) override
       {
          if (name == "t") { t = _value; return; }
       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {
          if (name == "t") return t;
       throw std::runtime_error("No such property");
       return 0.0;
       }

       void set_generic_function(std::string name, std::shared_ptr<dolfin::GenericFunction> _value) override
       {

       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {

       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_19d4f72bbb6f8de1c093e84b37795bdd()
{
  return new dolfin::dolfin_expression_19d4f72bbb6f8de1c093e84b37795bdd;
}

