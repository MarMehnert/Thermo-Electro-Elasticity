/* =========================
 * THERMO-ELECTRO-ELASTICITY
 * =========================
 * Problem description:
 *   Axial deformation (elongation) of a cylinder, with an electric
 *   field induced in the axial direction and a temperature gradient
 *   prescribed between the inner and outer radial surfaces.
 *
 * Initial implementation
 *   Author: Markus Mehnert (2015)
 *           Friedrich-Alexander University Erlangen-Nuremberg
 *   Description:
 *           Staggered one-way coupling of quasi-static
 *           linear-elasticity and thermal (conductivity) problems
 *
 * Extensions
 *   Author: Jean-Paul Pelteret (2015)
 *            Friedrich-Alexander University Erlangen-Nuremberg
 *   Description:
 *       [X] Nonlinear, finite deformation quasi-static  elasticity
 *       [X] Nonlinear quasi-static thermal (conductivity) problem
 *       [X] Nonlinear iterative solution scheme (Newton-Raphson)
 *           that encompasses staggered thermal / coupled EM
 *           solution update
 *       [X] Parallelisation via Trilinos (and possibly PETSc)
 *       [X] Parallel output of solution, residual
 *       [X] Choice of direct and indirect solvers
 *       [X] Adaptive grid refinement using Kelly error estimator
 *       [X] Parameter collection
 *       [X] Generic continuum point framework for integrating
 *           constitutive models
 *       [X] Nonlinear constitutive models
 *          [X] St. Venant Kirchoff
 *              + Materially linear thermal conductivity
 *          [X] Fully decoupled NeoHookean
 *              + Materially isotropic dielectric material
 *              + Spatially isotropic thermal conductivity
 *          [X] One-way coupled thermo-electro-mechanical model
 *              based on Markus' paper (Mehnert2015a)
 *              + Spatially isotropic thermal conductivity
 *
 *  References:
 *  Wriggers, P. Nonlinear finite element methods. 2008
 *  Holzapfel, G. A. Nonlinear solid mechanics. 2007
 *  Vu, K., On coupled BEM-FEM simulation of nonlinear electro-elastostatics
 */

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/iterative_inverse.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/block_sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>


#include <deal.II/lac/generic_linear_algebra.h>
#define USE_TRILINOS_LA
namespace LA
{
#ifdef USE_TRILINOS_LA
using namespace dealii::LinearAlgebraTrilinos;
#else
using namespace dealii::LinearAlgebraPETSc;
#endif
}

#include <mpi.h>
#include <fstream>
#include <iostream>

namespace Coupled_TEE
{
using namespace dealii;

template <int dim>
class BoundaryValues_XY : public Function<dim>
{
public:
	BoundaryValues_XY (const double neu, const unsigned int n_components)
: Function<dim>(n_components),
  lambda_i (neu)
  { }

	virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;
	virtual void vector_value (const Point<dim> &p,
			Vector<double>   &value) const;

private:

	const double lambda_i;
};
template <int dim>
double
BoundaryValues_XY<dim>::value (const Point<dim>  &p,
		const unsigned int component) const
		{
	Assert (component < this->n_components,
			ExcIndexRange (component, 0, this->n_components));

	if (component < dim)
	{
		return (lambda_i)*p[component];
	}
	else
		return 0.0;
		}

template <int dim>
void
BoundaryValues_XY<dim>::vector_value (const Point<dim> &p,
		Vector<double>   &values) const
		{
	for (unsigned int c=0; c<this->n_components; ++c)
		values(c) = BoundaryValues_XY<dim>::value (p, c);
		}



struct Parameters
{
	// Boundary ids
	// Boundary ids
	static constexpr unsigned int boundary_id_bottom = 0;
	static constexpr unsigned int boundary_id_top = 1;
	static constexpr unsigned int boundary_id_front = 2;
	static constexpr unsigned int boundary_id_back = 3;
	static constexpr unsigned int boundary_id_left = 4;
	static constexpr unsigned int boundary_id_right = 5;

	// Boundary conditions
	static constexpr double temperature_difference = 100.0;
	static constexpr double axial_displacement = 0.0;
	static constexpr double radial_displacement = 0.0;

	//    Potential difference in V
	static constexpr double potential_difference = 000;

	// Time
	static constexpr double dt = 0.1;
	static constexpr unsigned int n_timesteps = 10;

	//J-Ps additions

	static constexpr double time_end = 50.0e-3;
	static constexpr double time_delta = time_end/(static_cast<double>(n_timesteps));

	// Refinement
	static constexpr unsigned int n_global_refinements = 0;
	static constexpr bool perform_AMR = false;
	static constexpr unsigned int n_ts_per_refinement = 2;
	static constexpr unsigned int max_grid_level = 4;
	static constexpr double frac_refine = 0.3;
	static constexpr double frac_coarsen = 0.03;

	// Finite element
	static constexpr unsigned int poly_order =1;

	// Nonlinear solver
	static constexpr unsigned int max_newton_iterations = 20;
	static constexpr double max_res_T_norm = 1e-6;
	static constexpr double max_res_uV_norm = 1e-9;
	static constexpr double max_res_abs = 1e-6;

	// Linear solver: Thermal
	static const std::string solver_type_T;
	static constexpr double tol_rel_T = 1e-6;

	// Linear solver: Electro-mechanical
	static const std::string solver_type_EM;
	static constexpr double tol_rel_EM = 1e-6;
};
const std::string Parameters::solver_type_T = "Direct";
const std::string Parameters::solver_type_EM = "Direct";

namespace Material
{
struct Coefficients
{
	static constexpr double length_scale = 1.0;

	// Parameters in N, mm, V

	// Elastic parameters
	static constexpr double g_0 = 100.0e-3;// MPa=N/mm^2
	static constexpr double g_1 = -1e-6; // N/V^2

	// Electro parameters
	static constexpr double epsilon_0 =8.854187817e-12; // F/m = C/(V*m)= (A*s)/(V*m) = N/(V*V)
	static constexpr double c_1 = 0.2*epsilon_0;//in N/(V*V)				0.2* epsilon_0; //10; // in relation to electric permittivity epsilon
	static constexpr double c_2 = -2000.0*epsilon_0;//in N/(V*V)				6; // as in Vu


	// Independent of length and voltage units

	static constexpr double nu = 0.35; // Poisson ratio
	static constexpr double mu = g_0; // Small strain shear modulus
	static constexpr double lambda = 2.0*mu*nu/(1.0-2.0*nu); // Lame constant
	static constexpr double kappa = 2.0*mu*(1.0+nu)/(3.0*(1.0-2.0*nu)); // mu = 3*10^5 Pa

	// Thermal parameters
	static constexpr double c_0 = 460e6; // specific heat capacity in J/(kg*K)
	static constexpr double alpha = 20e-6; //thermal expansion coefficient in 1/K
	static constexpr double theta_0 = 293; // in K
	static constexpr double k = 0.50; // Heat conductivity in N/(s*K)


};

template<int dim>
struct Values
{
	Values (const Tensor<2,dim> & F,
			const Tensor<1,dim> & E,
			const Tensor<1,dim> & R,
			const double theta)
	: F (F),
	  E (E),
	  R (R),
	  theta (theta),

	  C (symmetrize(transpose(F)*F)),
	  b (symmetrize(F*transpose(F))),
	  GLST (0.5*(C - unit_symmetric_tensor<dim>())),
	  C_inv (symmetrize(invert(static_cast<Tensor<2,dim> >(C)))),
	  J (determinant(F)),

	  I1 (first_invariant(C)), // tr(C)
	  I4 (E*E), // [ExE].I
	  I5 (E*(C_inv*E)) // [ExE].C^{-1}
	{}

	// Directly related to solution field
	const Tensor<2,dim> F; // Deformation gradient
	const Tensor<1,dim> E; // Electric field vector
	const Tensor<1,dim> R; // Negative thermal gradient vector
	const double theta; // Temperature

	// Commonly used elastic quantities
	const SymmetricTensor<2,dim> C; // Right Cauchy-Green deformation tensor
	const SymmetricTensor<2,dim> b; // left Cauchy-Green deformation tensor
	const SymmetricTensor<2,dim> GLST; // Green-Lagrange strain tensor
	const SymmetricTensor<2,dim> C_inv;
	const double J;

	// Invariants
	const double I1;
	const double I4;
	const double I5;

	// === First derivatives ===
	// --- Mechanical ---
	SymmetricTensor<2,dim>
	dI1_dC () const
	{
		return unit_symmetric_tensor<dim>();
	}

	SymmetricTensor<2,dim>
	dI4_dC () const
	{
		return SymmetricTensor<2,dim>();
	}

	SymmetricTensor<2,dim>
	dI5_dC () const
	{
		const SymmetricTensor<2,dim> ExE = outer_product(E,E);
		return ExE*dC_inv_dC();
	}

	SymmetricTensor<2,dim>
	dJ_dC () const
	{
		return (0.5*J)*C_inv;
	}

	// --- Electric ---
	Tensor<1,dim>
	dI4_dE () const
	{
		return 2.0*E;
	}

	Tensor<1,dim>
	dI5_dE () const
	{
		return 2.0*(C_inv*E);
	}

	// === Second derivatives ===
	// --- Mechanical ---
	SymmetricTensor<4,dim>
	d2I1_dC_dC () const
	{
		return SymmetricTensor<4,dim>();
	}

	SymmetricTensor<4,dim>
	d2I4_dC_dC () const
	{
		return SymmetricTensor<4,dim>();
	}

	SymmetricTensor<4,dim>
	d2I5_dC_dC () const
	{
		const Tensor<1,dim> Y = C_inv*E;

		SymmetricTensor<4,dim> d2I5_dC_dC;
		for (unsigned int A=0; A<dim; ++A)
			for (unsigned int B=A; B<dim; ++B)
				for (unsigned int C=0; C<dim; ++C)
					for (unsigned int D=C; D<dim; ++D)
					{
						// Need to ensure symmetries of (I,J) and (K,L)
						d2I5_dC_dC[A][B][C][D] += 0.25*( Y[A]*Y[C]*C_inv[B][D] + Y[B]*Y[C]*C_inv[A][D] + Y[A]*Y[D]*C_inv[B][C] + Y[B]*Y[D]*C_inv[A][C]
						                                                                                                                            + Y[C]*Y[B]*C_inv[A][D] + Y[C]*Y[A]*C_inv[B][D] + Y[D]*Y[B]*C_inv[A][C] + Y[D]*Y[A]*C_inv[B][C] );
					}

		return d2I5_dC_dC;
	}

	SymmetricTensor<4,dim>
	dC_inv_dC () const
	{
		SymmetricTensor<4,dim> dC_inv_dC;

		for (unsigned int A=0; A<dim; ++A)
			for (unsigned int B=A; B<dim; ++B)
				for (unsigned int C=0; C<dim; ++C)
					for (unsigned int D=C; D<dim; ++D)
						dC_inv_dC[A][B][C][D] = -0.5*(C_inv[A][C]*C_inv[B][D]+ C_inv[A][D]*C_inv[B][C]);

		return dC_inv_dC;
	}

	// --- Electric ---
	SymmetricTensor<2,dim>
	d2I4_dE_dE () const
	{
		return 2.0*unit_symmetric_tensor<dim>();
	}

	SymmetricTensor<2,dim>
	d2I5_dE_dE () const
	{
		return 2.0*C_inv;
	}

	// --- Electro-mechanical ---
	Tensor<3,dim>
	d2I4_dE_dC () const
	{
		return Tensor<3,dim>();
	}

	Tensor<3,dim>
	d2I5_dE_dC () const
	{
		const Tensor<1,dim> Y = C_inv*E;

		Tensor<3,dim> d2I5_dE_dC;
		for (unsigned int A=0; A<dim; ++A)
			for (unsigned int B=0; B<dim; ++B)
				for (unsigned int C=0; C<dim; ++C)
					d2I5_dE_dC[A][B][C] -= C_inv[C][A]*Y[B] + C_inv[C][B]*Y[A];

		return d2I5_dE_dC;
	}

};

template<int dim>
struct CM_Base
{
	CM_Base (const Tensor<2,dim> & F,
			const Tensor<1,dim> & E,
			const Tensor<1,dim> & R,
			const double theta)
	: values (F,E,R,theta)
	{}

	virtual ~CM_Base () {}

	// --- Kinematic Quantities ---

	const Values<dim> values;

	// --- Kinetic Quantities ---

	// Second Piola-Kirchhoff stress tensor
	inline SymmetricTensor<2,dim>
	get_S () const
	{
		return 2.0*get_dPsi_dC();
	}

	// Referential electric displacement vector
	inline Tensor<1,dim>
	get_D () const
	{
		return -get_dPsi_dE();
	}

	// Referential thermal flux vector
	inline Tensor<1,dim>
	get_Q () const
	{
		// Q = -K.Grad_T
		return get_K()*values.R;
	}

	// --- Material tangents ---

	// Referential elastic stiffness tensor
	inline SymmetricTensor<4,dim>
	get_C () const
	{
		return 4*get_d2Psi_dC_dC();
	}

	// Referential pizeo-elasticity tensor
	inline Tensor<3,dim>
	get_P () const
	{
		return -2.0*get_d2Psi_dE_dC();
	}

	// Referential dielectric tensor
	inline SymmetricTensor<2,dim>
	get_DD () const
	{
		return -get_d2Psi_dE_dE();
	}

	// Referential thermal flux vector
	inline SymmetricTensor<2,dim>
	get_K () const
	{
		return get_d2Psi_dR_dR();
	}

protected:

	// --- Total stored energy ---

	virtual double
	get_Psi (void) const = 0;

	// --- Pure mechanical volumetric response ---

	double
	get_W_J (const double kappa,
			const double g_0) const
	{
		const double &J = values.J;

		// See Wriggers p45 equ 3.118
		return (kappa/2)*(std::log(J)*std::log(J)) // This part is from Holzapfel p
				- g_0*std::log(J); // This part cancels out dilatory response of Neo-Hookean material
	}

	// Derivative of the volumetric free energy with respect to
	// $\widetilde{J}$ return $\frac{\partial
	// \Psi_{\text{vol}}(\widetilde{J})}{\partial \widetilde{J}}$
	double
	get_dW_J_dJ (const double kappa,
			const double g_0) const
	{
		const double tr_C=values.C[0][0]+values.C[1][1]+values.C[2][2];
		const double &J = values.J;
		return ((kappa)*(std::log(J))/J
				- g_0/J);
	}


	// Second derivative of the volumetric free energy wrt $\widetilde{J}$. We
	// need the following computation explicitly in the tangent so we make it
	// public.  We calculate $\frac{\partial^2
	// \Psi_{\textrm{vol}}(\widetilde{J})}{\partial \widetilde{J} \partial
	// \widetilde{J}}$
	double
	get_d2W_J_dJ_dJ (const double kappa,
			const double g_0) const
	{
		const double &J = values.J;
		return (kappa)*(1-std::log(J))/(J*J)
				+ g_0/(J*J);
	}


	SymmetricTensor<2,dim>
	get_dW_J_dC (const double kappa,
			const double g_0) const
			{
		// See Wriggers p46 eqs. 3.123, 3.124; Holzapfel p230
		return get_dW_J_dJ(kappa, g_0)*this->values.dJ_dC();
			}

	SymmetricTensor<4,dim>
	get_d2W_J_dC_dC (const double kappa,
			const double g_0) const
			{
		const double &J = values.J;
		const SymmetricTensor<2,dim> &C_inv = values.C_inv;

		// See Holzapfel p255, p265 ; Wriggers p75-76
		const double dW_J_dJ = get_dW_J_dJ(kappa, g_0);
		const double d2W_J_dJ_dJ = get_d2W_J_dJ_dJ(kappa, g_0);

		const SymmetricTensor<4,dim> dC_inv_dC = this->values.dC_inv_dC();

		SymmetricTensor<4,dim> d2W_J_dC_dC;
		const double  p_tilde = dW_J_dJ + J*d2W_J_dJ_dJ;
		const double &p = dW_J_dJ;
		for (unsigned int A=0; A<dim; ++A)
			for (unsigned int B=A; B<dim; ++B)
				for (unsigned int C=0; C<dim; ++C)
					for (unsigned int D=C; D<dim; ++D)
					{
						d2W_J_dC_dC[A][B][C][D]
						                     = 0.25*J*( p_tilde*(C_inv[A][B]*C_inv[C][D])
						                    		 + (2.0*p)*dC_inv_dC[A][B][C][D] );
					}

		return d2W_J_dC_dC;
			}

	// --- Coupled mechanical response ---

	virtual SymmetricTensor<2,dim>
	get_dPsi_dC () const = 0;

	virtual SymmetricTensor<4,dim>
	get_d2Psi_dC_dC () const = 0;


	// --- Coupled electric response ---

	virtual Tensor<1,dim>
	get_dPsi_dE () const = 0;

	virtual Tensor<1,dim>
	get_d2Psi_dE_dtheta () const = 0;

	virtual SymmetricTensor<2,dim>
	get_d2Psi_dE_dE () const = 0;

	// --- Electro-mechanical coupling ---

	virtual Tensor<3,dim>
	get_d2Psi_dE_dC () const = 0;

	// --- (Weakly coupled) Thermal response ---

	virtual SymmetricTensor<2,dim>
	get_d2Psi_dR_dR () const = 0;
};

template<int dim>
struct CM_NeoHookean : public CM_Base<dim>
{
	CM_NeoHookean (const Tensor<2,dim> &F,
			const Tensor<1,dim> &E,
			const Tensor<1,dim> &R,
			const double theta)
	: CM_Base<dim> (F,E,R,theta)
	  {}

	virtual ~CM_NeoHookean () {}

protected:

	// --- Total stored energy ---
	virtual double
	get_Psi (void) const
	{

		return get_W_F(Coefficients::lambda,
				Coefficients::mu); // Pure mechanical response
	}

	// === ELECTRO-MECHANICAL PROBLEM ===
	inline double
	get_W_F (const double lambda,
			const double mu) const
	{
		const double &I1 = this->values.I1;
		const double &J = this->values.J;
		return (mu/2)*(I1-3)-mu*std::log(J) + lambda/2*(std::log(J)*std::log(J));
	}

	// --- Mechanical contributions ---
	virtual SymmetricTensor<2,dim>
	get_dPsi_dC () const
	{
		return get_dW_F_dC(Coefficients::lambda,
				Coefficients::mu); // Pure mechanical response
	}

	virtual double
	theta_rat () const
	{
		return 0;
	}
	inline SymmetricTensor<2,dim>
	get_dW_F_dC (const double lambda,
			const double mu) const
			{
		const double &J = this->values.J;
		const SymmetricTensor<2,dim> &C_inv = this->values.C_inv;

		return mu*(unit_symmetric_tensor<dim>()-C_inv) + lambda*std::log(J)*C_inv;
			}

	virtual SymmetricTensor<4,dim>
	get_d2Psi_dC_dC () const
	{
		return get_d2W_F_dC_dC(Coefficients::lambda,
				Coefficients::mu);
	}

	inline SymmetricTensor<4,dim>
	get_d2W_F_dC_dC (const double lambda,
			const double mu) const
			{
		const double &J = this->values.J;
		const SymmetricTensor<2,dim> &C_inv = this->values.C_inv;
		static const SymmetricTensor<4,dim> C_invxC_inv = outer_product(C_inv,
				C_inv);
		const SymmetricTensor<4,dim> dC_inv_dC = this->values.dC_inv_dC();

		return (-2*mu+2*lambda*std::log(J))*dC_inv_dC + lambda*C_invxC_inv;
			}

	// --- Electric component ---
	virtual Tensor<1,dim>
	get_dPsi_dE () const
	{
		return Tensor<1,dim> ();
	}

	virtual Tensor<1,dim>
	get_d2Psi_dE_dtheta () const
	{
		return Tensor<1,dim> ();
	}

	virtual SymmetricTensor<2,dim>
	get_d2Psi_dE_dE () const
	{
		return unit_symmetric_tensor<dim>();
	}

	// --- Electro-mechanical coupling ---
	virtual Tensor<3,dim>
	get_d2Psi_dE_dC () const
	{
		return Tensor<3,dim> ();
	}

	// --- Thermal component ---
	virtual SymmetricTensor<2,dim>
	get_d2Psi_dR_dR () const
	{
		return unit_symmetric_tensor<dim>();
	}
};



template<int dim>
struct CM_St_Venant_Kirchhoff : public CM_Base<dim>
{
	CM_St_Venant_Kirchhoff (const Tensor<2,dim> &F,
			const Tensor<1,dim> &E,
			const Tensor<1,dim> &R,
			const double theta)
	: CM_Base<dim> (F,E,R,theta)
	  {}

	virtual ~CM_St_Venant_Kirchhoff () {}

protected:

	// --- Total stored energy ---
	virtual double
	get_Psi (void) const
	{
		return get_W_F(Coefficients::lambda,
				Coefficients::mu); // Pure mechanical response
	}

	// --- Purely mechanical component ---
	virtual SymmetricTensor<2,dim>
	get_dPsi_dC () const
	{
		return get_dW_F_dC(Coefficients::lambda,
				Coefficients::mu); // Pure mechanical response
	}

	virtual SymmetricTensor<4,dim>
	get_d2Psi_dC_dC () const
	{
		return get_d2W_F_dC_dC(Coefficients::lambda,
				Coefficients::mu); // Pure mechanical response
	}

	inline double
	get_W_F (const double lambda,
			const double mu) const
	{
		const SymmetricTensor<2,dim> &E = this->values.GLST;
		const double trace_E2 = E*E;

		return lambda/2.0*trace(E) + mu*trace_E2;
	}

	inline SymmetricTensor<2,dim>
	get_dW_F_dC (const double lambda,
			const double mu) const
			{
		static const SymmetricTensor<2,dim> I = unit_symmetric_tensor<dim>();
		const SymmetricTensor<2,dim> &E = this->values.GLST;

		return 0.5*(2.0*mu*E + lambda*trace(E)*I);
			}

	inline SymmetricTensor<4,dim>
	get_d2W_F_dC_dC (const double lambda,
			const double mu) const
			{
		static const SymmetricTensor<4,dim> IxI = outer_product(unit_symmetric_tensor<dim>(),
				unit_symmetric_tensor<dim>());
		static const SymmetricTensor<4,dim> II  = identity_tensor<dim>();

		return 0.25*(2.0*mu*II + lambda*IxI);
			}

	// --- Electric component ---
	virtual Tensor<1,dim>
	get_dPsi_dE () const
	{
		return Tensor<1,dim> ();
	}

	virtual SymmetricTensor<2,dim>
	get_d2Psi_dE_dE () const
	{
		return unit_symmetric_tensor<dim>();
	}

	// --- Electro-mechanical coupling ---
	virtual Tensor<3,dim>
	get_d2Psi_dE_dC () const
	{
		return Tensor<3,dim> ();
	}

	// --- Thermal component ---
	virtual SymmetricTensor<2,dim>
	get_d2Psi_dR_dR () const
	{
		return unit_symmetric_tensor<dim>();
	}
};

template<int dim>
struct CM_Decoupled_NeoHookean : public CM_Base<dim>
{
	CM_Decoupled_NeoHookean  (const Tensor<2,dim> &F,
			const Tensor<1,dim> &E,
			const Tensor<1,dim> &R,
			const double theta)
	: CM_Base<dim> (F,E,R,theta)
	  {}

	virtual ~CM_Decoupled_NeoHookean () {}

protected:

	// --- Total stored energy ---
	virtual double
	get_Psi (void) const
	{
		return this->get_W_J(Coefficients::kappa,
				Coefficients::mu) // Volumetric response W_J = W_J(J)
				+ get_W_F(Coefficients::mu) // Mechanical response W = W(F)
				+ get_W_E(Coefficients::c_1); // Electric response W = W(E)
	}

	// --- Mechanical component ---
	virtual SymmetricTensor<2,dim>
	get_dPsi_dC () const
	{
		return this->get_dW_J_dC(Coefficients::kappa,
				Coefficients::mu) // Volumetric response W_J = W_J(J)
				+ get_dW_F_dC(Coefficients::mu); // Mechanical response W = W(F)
	}

	virtual SymmetricTensor<4,dim>
	get_d2Psi_dC_dC () const
	{
		return this->get_d2W_J_dC_dC(Coefficients::kappa,
				Coefficients::mu) // Volumetric response W_J = W_J(J)
				+ get_d2W_F_dC_dC(Coefficients::mu); // Mechanical response W = W(F)
	}

	inline double
	get_W_F (const double mu) const
	{
		const double &I1 = this->values.I1;
		return mu/2.0*(I1-dim);
	}

	inline SymmetricTensor<2,dim>
	get_dW_F_dC (const double mu) const
	{
		return (mu/2.0)*unit_symmetric_tensor<dim>();
	}

	inline SymmetricTensor<4,dim>
	get_d2W_F_dC_dC (const double mu) const
	{
		return SymmetricTensor<4,dim>();
	}

	// --- Electric component ---
	virtual Tensor<1,dim>
	get_dPsi_dE () const
	{
		return get_dW_E_dE(Coefficients::c_1); // Electric response W = W(E)
	}

	virtual SymmetricTensor<2,dim>
	get_d2Psi_dE_dE () const
	{
		return get_d2W_E_dE_dE(Coefficients::c_1); // Electric response W = W(E)
	}

	inline double
	get_W_E (const double c_1) const
	{
		return c_1*(this->values.E*this->values.E);
	}

	inline Tensor<1,dim>
	get_dW_E_dE (const double c_1) const
	{
		return 2.0*(c_1*this->values.E);
	}

	inline SymmetricTensor<2,dim>
	get_d2W_E_dE_dE (const double c_1) const
	{
		return 2.0*(c_1*unit_symmetric_tensor<dim>());
	}

	// --- Electro-mechanical coupling ---
	virtual Tensor<3,dim>
	get_d2Psi_dE_dC () const
	{
		return Tensor<3,dim> ();
	}

	// --- Thermal component ---
	virtual SymmetricTensor<2,dim>
	get_d2Psi_dR_dR () const
	{
		return get_k(Coefficients::k);
	}

	inline SymmetricTensor<2,dim>
	get_k (const double k) const
	{
		// Spatially linear conductivity
		const Tensor<2,dim> F_inv = invert(this->values.F);
		return k*symmetrize(F_inv*transpose(F_inv));
	}
};

template<int dim>
struct CM_Weakly_Coupled_TEE : public CM_Base<dim>
{
	CM_Weakly_Coupled_TEE  (const Tensor<2,dim> &F,
			const Tensor<1,dim> &E,
			const Tensor<1,dim> &R,
			const double theta)
	: CM_Base<dim> (F,E,R,theta)
	  {}

	virtual ~CM_Weakly_Coupled_TEE () {}

protected:

	// --- Total stored energy ---
	virtual double
	get_Psi (void) const
	{
		const double W_J = this->get_W_J(Coefficients::kappa,
				Coefficients::mu); // Volumetric response W_J = W_J(J)
		const double W_FE = get_W_FE(Coefficients::g_0,
				Coefficients::g_1,
				Coefficients::c_1,
				Coefficients::c_2); // Electro-mechanical response W = W(F,E)
		const double M_J = get_M_J(Coefficients::kappa,
				Coefficients::alpha); // Thermal dilatory response M = M(J)
		const double c_T = get_c_T(Coefficients::c_0); // Thermal response c = C(T)

		// Mehnert2015a equ. 46
		return theta_ratio()*(W_J + W_FE)
				- theta_difference()*M_J
				+ c_T;
	}

	// === ELECTRO-MECHANICAL PROBLEM ===
	inline double
	get_W_FE (const double g_0,
			const double g_1,
			const double c_1,
			const double c_2) const
	{
		const double &I1 = this->values.I1;
		const double &I4 = this->values.I4;
		const double &I5 = this->values.I5;
		const double &theta=this->values.theta;
		return (g_0 + g_1*I4)/2.0*(I1-dim)
				+ c_1*I4
				+ c_2*I5;
	}

	// --- Mechanical contributions ---
	virtual SymmetricTensor<2,dim>
	get_dPsi_dC () const
	{
		return theta_ratio()*(this->get_dW_J_dC(Coefficients::kappa,
				Coefficients::mu) // Volumetric response W_J = W_J(J)
				+ get_dW_FE_dC(Coefficients::g_0,
						Coefficients::g_1,
						Coefficients::c_1,
						Coefficients::c_2)) // Electro-mechanical response W = W(F,E)
						- theta_difference()*get_dM_J_dC(Coefficients::kappa,
								Coefficients::alpha); // Thermal dilatory response M = M(J)
	}

	inline SymmetricTensor<2,dim>
	get_dW_FE_dC (const double g_0,
			const double g_1,
			const double c_1,
			const double c_2) const
			{
		const double &I1 = this->values.I1;
		const double &I4 = this->values.I4;
		const double &theta = this->values.theta;
		return (g_0 + g_1*I4)/2.0*this->values.dI1_dC()
				+ (c_1 + g_1/2.0*(I1-dim))*this->values.dI4_dC()
				+ c_2*this->values.dI5_dC();
			}


	virtual SymmetricTensor<4,dim>
	get_d2Psi_dC_dC () const
	{
		return theta_ratio()*(this->get_d2W_J_dC_dC(Coefficients::kappa,
				Coefficients::mu) // Volumetric response W_J = W_J(J)
				+ get_d2W_FE_dC_dC(Coefficients::g_0,
						Coefficients::g_1,
						Coefficients::c_1,
						Coefficients::c_2)) // Electro-mechanical response W = W(F,E)
						- theta_difference()*get_d2M_J_dC_dC(Coefficients::kappa,
								Coefficients::alpha); // Thermal dilatory response M = M(J)
	}


	inline SymmetricTensor<4,dim>
	get_d2W_FE_dC_dC (const double g_0,
			const double g_1,
			const double c_1,
			const double c_2) const
			{
		const double &I1 = this->values.I1;
		const double &I4 = this->values.I4;
		return (g_0 + g_1*I4)/2.0*this->values.d2I1_dC_dC()
				+ (c_1 + g_1/2.0*(I1-dim))*this->values.d2I4_dC_dC()
				+ (g_1/2.0)*outer_product(this->values.dI4_dC(), this->values.dI1_dC())
				+ c_2*this->values.d2I5_dC_dC();
			}

	// --- Electric contributions ---
	virtual Tensor<1,dim>
	get_dPsi_dE () const
	{
		return theta_ratio()*get_dW_FE_dE(Coefficients::g_0,
				Coefficients::g_1,
				Coefficients::c_1,
				Coefficients::c_2); // Electro-mechanical response W = W(F,E)
	}

	virtual Tensor<1,dim>
	get_d2Psi_dE_dtheta () const
	{
		return get_dW_FE_dE(Coefficients::g_0,
				Coefficients::g_1,
				Coefficients::c_1,
				Coefficients::c_2)/Coefficients::theta_0; // Electro-mechanical response W = W(F,E)
	}

	inline Tensor<1,dim>
	get_dW_FE_dE (const double g_0,
			const double g_1,
			const double c_1,
			const double c_2) const
			{
		const double &I1 = this->values.I1;
		return (c_1 + (g_1/2.0)*(I1-dim))*this->values.dI4_dE()
				+ c_2*this->values.dI5_dE();
			}

	virtual SymmetricTensor<2,dim>
	get_d2Psi_dE_dE () const
	{
		return theta_ratio()*get_d2W_FE_dE_dE(Coefficients::g_0,
				Coefficients::g_1,
				Coefficients::c_1,
				Coefficients::c_2); // Electro-mechanical response W = W(F,E)
	}

	inline SymmetricTensor<2,dim>
	get_d2W_FE_dE_dE (const double g_0,
			const double g_1,
			const double c_1,
			const double c_2) const
			{
		const double &I1 = this->values.I1;
		return (c_1 + (g_1/2.0)*(I1-dim))*this->values.d2I4_dE_dE()
				+ c_2*this->values.d2I5_dE_dE();
			}

	// --- Electro-mechanical coupling ---
	virtual Tensor<3,dim>
	get_d2Psi_dE_dC () const
	{
		return theta_ratio()*get_dW_FE_dE_dC(Coefficients::g_0,
				Coefficients::g_1,
				Coefficients::c_1,
				Coefficients::c_2); // Electro-mechanical response W = W(F,E)
	}

	inline Tensor<3,dim>
	get_dW_FE_dE_dC (const double g_0,
			const double g_1,
			const double c_1,
			const double c_2) const
			{
		const double &I1 = this->values.I1;


		return (c_1 + g_1/2.0*(I1-dim))*this->values.d2I4_dE_dC()
				/*+g_1/2.0*dI4_dE*this->values.dI1_dC()*/
				+ c_2*this->values.d2I5_dE_dC();
			}

	// --- Thermo-mechanical component ---
	inline double
	get_M_J (const double kappa,
			const double alpha) const
	{
		// Mehnert2015a equ. 42a; Holzapfel p338 equ. 7.85
		const double &J = this->values.J;
		return dim*kappa*alpha*std::log(J)/J;
	}

	inline double
	get_dM_J_dJ (const double kappa,
			const double alpha) const
	{
		const double &J = this->values.J;
		return dim*kappa*alpha*(1-std::log(J))/(J*J);
	}


	inline double
	get_d2M_J_dJ_dJ (const double kappa,
			const double alpha) const
	{
		const double &J = this->values.J;
		return dim*kappa*alpha*(2*std::log(J)-3)/(J*J*J);
	}

	SymmetricTensor<2,dim>
	get_dM_J_dC (const double kappa,
			const double alpha) const
			{
		// See Wriggers p46 eqs. 3.123, 3.124; Holzapfel p230
		return get_dM_J_dJ(kappa, alpha)*this->values.dJ_dC();
			}

	SymmetricTensor<4,dim>
	get_d2M_J_dC_dC (const double kappa,
			const double alpha) const
			{
		const double &J = this->values.J;
		const SymmetricTensor<2,dim> &C_inv = this->values.C_inv;

		// See Holzapfel p255, p265 ; Wriggers p75-76
		const double dM_J_dJ = get_dM_J_dJ(kappa, alpha);
		const double d2M_J_dJ_dJ = get_d2M_J_dJ_dJ(kappa, alpha);

		const SymmetricTensor<4,dim> dC_inv_dC = this->values.dC_inv_dC();

		SymmetricTensor<4,dim> d2M_J_dC_dC;
		const double  m_tilde = dM_J_dJ + J*d2M_J_dJ_dJ;
		const double &m = dM_J_dJ;
		for (unsigned int A=0; A<dim; ++A)
			for (unsigned int B=A; B<dim; ++B)
				for (unsigned int C=0; C<dim; ++C)
					for (unsigned int D=C; D<dim; ++D)
					{
						d2M_J_dC_dC[A][B][C][D]
						                     = 0.25*J*( m_tilde*(C_inv[A][B]*C_inv[C][D])
						                    		 + (2.0*m)*dC_inv_dC[A][B][C][D] );
					}

		return d2M_J_dC_dC;
			}

	// --- Thermal component ---
	double
	get_c_T (const double c_0) const
	{
		// Mehnert2015a equ. 46
		return c_0*(theta_difference() - this->values.theta*std::log(theta_ratio()));
	}

	// === THERMAL PROBLEM ===
	virtual SymmetricTensor<2,dim>
	get_d2Psi_dR_dR () const
	{
		return get_k(Coefficients::k);
	}

	inline SymmetricTensor<2,dim>
	get_k (const double k) const
	{
		// Spatially linear conductivity
		const Tensor<2,dim> F_inv = invert(this->values.F);
		return k*symmetrize(F_inv*transpose(F_inv));
	}

	// === HELPER FUNCTIONS ===
	double
	theta_ratio () const
	{
		return this->values.theta/Coefficients::theta_0;
	}

	double
	theta_difference () const
	{
		return this->values.theta - Coefficients::theta_0;
	}
};

}

template<int dim>
class CoupledProblem
{
	//      typedef Material::CM_St_Venant_Kirchhoff<dim> Continuum_Point;
	//      typedef Material::CM_Decoupled_NeoHookean<dim> Continuum_Point;
	typedef Material::CM_Weakly_Coupled_TEE<dim> Continuum_Point;


public:
	CoupledProblem ();
	~CoupledProblem ();
	void
	run ();

private:
	void
	make_grid ();
	void
	refine_grid ();
	void
	setup_system ();
	void
	make_constraints (const unsigned int newton_iteration, const unsigned int timestep);
	void
	assemble_system_thermo (const unsigned int newton_iteration, const unsigned int ts);
	void
	solve_thermo (LA::MPI::BlockVector & solution_update);
	void
	assemble_system_mech (const unsigned int ts);
	void
	solve_mech (LA::MPI::BlockVector & solution_update);
	void
	solve_nonlinear_timestep (const double time, const int ts);
	void
	output_results (const unsigned int timestep) const;

	const unsigned int n_blocks;
	const unsigned int first_u_component; // Displacement
	const unsigned int V_component; // Voltage / Potential difference
	const unsigned int T_component; // Temperature
	const unsigned int n_components;

	enum
	{
		uV_block = 0,
		T_block  = 1
	};

	enum
	{
		u_dof = 0,
		V_dof = 1,
		T_dof = 2
	};

	const FEValuesExtractors::Vector displacement;
	const FEValuesExtractors::Scalar x_displacement;
	const FEValuesExtractors::Scalar y_displacement;
	const FEValuesExtractors::Scalar z_displacement;
	const FEValuesExtractors::Scalar voltage;
	const FEValuesExtractors::Scalar temperature;

	MPI_Comm           mpi_communicator;
	const unsigned int n_mpi_processes;
	const unsigned int this_mpi_process;
	mutable ConditionalOStream pcout;
	mutable TimerOutput computing_timer;

	parallel::distributed::Triangulation<dim> triangulation;
	DoFHandler<dim> dof_handler;

	IndexSet locally_owned_dofs;
	IndexSet locally_relevant_dofs;
	std::vector<IndexSet> locally_owned_partitioning;
	std::vector<IndexSet> locally_relevant_partitioning;

	const unsigned int poly_order;
	FESystem<dim> fe_cell;
	FESystem<dim> fe_face;

	QGauss<dim> qf_cell;
	QGauss<dim-1> qf_face;

	ConstraintMatrix hanging_node_constraints;
	ConstraintMatrix dirichlet_constraints;
	ConstraintMatrix all_constraints;

	LA::MPI::BlockSparseMatrix system_matrix;
	LA::MPI::BlockVector       system_rhs;
	LA::MPI::BlockVector       solution;

	LA::MPI::BlockVector locally_relevant_solution;
	LA::MPI::BlockVector locally_relevant_solution_t1;

};

template<int dim>
CoupledProblem<dim>::CoupledProblem ()
:
n_blocks (2),
first_u_component (0), // Displacement
V_component (first_u_component + dim), // Voltage / Potential difference
T_component (V_component+1), // Temperature
n_components (T_component+1),

displacement(first_u_component),
x_displacement(first_u_component),
y_displacement(first_u_component+1),
z_displacement(dim==3 ? first_u_component+2 : first_u_component+1),
voltage(V_component),
temperature(T_component),

mpi_communicator (MPI_COMM_WORLD),
n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
pcout(std::cout, this_mpi_process == 0),
computing_timer(mpi_communicator,
		pcout,
		TimerOutput::summary,
		TimerOutput::wall_times),

		triangulation (mpi_communicator,
				typename Triangulation<dim>::MeshSmoothing
				(Triangulation<dim>::smoothing_on_refinement |
						Triangulation<dim>::smoothing_on_coarsening)),
						dof_handler(triangulation),

						poly_order (Parameters::poly_order),
						fe_cell(FE_Q<dim> (poly_order), dim, // Displacement
								FE_Q<dim> (poly_order), 1, // Voltage
								FE_Q<dim> (poly_order), 1), // Temperature
								fe_face(FE_Q<dim> (poly_order), dim, // Displacement
										FE_Q<dim> (poly_order), 1, // Voltage
										FE_Q<dim> (poly_order), 1), // Temperature

										qf_cell(poly_order+1),
										qf_face(poly_order+1)
										{
										}

template<int dim>
CoupledProblem<dim>::~CoupledProblem ()
{
	dof_handler.clear();
}

template<int dim>
void
CoupledProblem<dim>::make_grid () //Generate thick walled cylinder
{
	TimerOutput::Scope timer_scope (computing_timer, "Make grid");


	GridIn<dim> grid_in;
	grid_in.attach_triangulation (triangulation);
	std::ifstream input_file("Cube_mm.inp");

	grid_in.read_abaqus (input_file);

	typename Triangulation<dim>::active_cell_iterator
	cell = triangulation.begin_active(),
	endc = triangulation.end();
	for (; cell != endc; ++cell)
	{
		for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
		{
			if (cell->face(f)->at_boundary())
			{
				const Point<dim> face_center = cell->face(f)->center();
				if (face_center[2] == 0)
				{ // Faces at cylinder bottom
					cell->face(f)->set_boundary_id(Parameters::boundary_id_bottom);
					cell->set_all_manifold_ids(0);
				}
				else if (face_center[2] == 1)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_top);
					cell->set_all_manifold_ids(0);
				}
				else if (face_center[1] == 0)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_left);
					cell->set_all_manifold_ids(0);
				}
				else if (face_center[1] == 1)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_right);
					cell->set_all_manifold_ids(0);
				}
				else if (face_center[0] == 0)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_front);
					cell->set_all_manifold_ids(0);
				}
				else if (face_center[0] == 1)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_back);
					cell->set_all_manifold_ids(0);
				}


			}
		}
	}

}

template<int dim>
void
CoupledProblem<dim>::setup_system ()
{
	TimerOutput::Scope timer_scope (computing_timer, "System setup");
	pcout << "Setting up the thermo-electro-mechanical system..." << std::endl;

	dof_handler.distribute_dofs(fe_cell);

	std::vector<types::global_dof_index>  block_component(n_components, uV_block); // Displacement
	block_component[V_component] = uV_block; // Voltage
	block_component[T_component] = T_block; // Temperature

	DoFRenumbering::Cuthill_McKee(dof_handler);
	DoFRenumbering::component_wise(dof_handler, block_component);

	std::vector<types::global_dof_index> dofs_per_block(n_blocks);
	DoFTools::count_dofs_per_block(dof_handler, dofs_per_block, block_component);
	const types::global_dof_index &n_u_V = dofs_per_block[0];
	const types::global_dof_index &n_th = dofs_per_block[1];

	pcout
	<< "Number of active cells: "
	<< triangulation.n_active_cells()
	<< std::endl
	<< "Total number of cells: "
	<< triangulation.n_cells()
	<< std::endl
	<< "Number of degrees of freedom: "
	<< dof_handler.n_dofs()
	<< " (" << n_u_V << '+' << n_th << ')'
	<< std::endl;

	locally_owned_partitioning.clear();
	locally_owned_dofs = dof_handler.locally_owned_dofs ();
	locally_owned_partitioning.push_back(locally_owned_dofs.get_view(0, n_u_V));
	locally_owned_partitioning.push_back(locally_owned_dofs.get_view(n_u_V, n_u_V+n_th));

	DoFTools::extract_locally_relevant_dofs (dof_handler,
			locally_relevant_dofs);
	locally_relevant_partitioning.clear();
	locally_relevant_partitioning.push_back(locally_relevant_dofs.get_view(0, n_u_V));
	locally_relevant_partitioning.push_back(locally_relevant_dofs.get_view(n_u_V, n_u_V+n_th));

	hanging_node_constraints.clear();
	hanging_node_constraints.reinit (locally_relevant_dofs);
	DoFTools::make_hanging_node_constraints(dof_handler,
			hanging_node_constraints);
	hanging_node_constraints.close();

	Table<2, DoFTools::Coupling> coupling(n_components, n_components);
	for (unsigned int ii = 0; ii < n_components; ++ii)
		for (unsigned int jj = 0; jj < n_components; ++jj)
			if (((ii < T_component) && (jj == T_component))
					|| ((ii == T_component) && (jj < T_component)))
				coupling[ii][jj] = DoFTools::none;
			else
				coupling[ii][jj] = DoFTools::always;

	TrilinosWrappers::BlockSparsityPattern sp (locally_owned_partitioning,
			locally_owned_partitioning,
			locally_relevant_partitioning,
			mpi_communicator);

	DoFTools::make_sparsity_pattern (dof_handler,
			coupling, sp,
			hanging_node_constraints,
			false,
			this_mpi_process);
	sp.compress();
	system_matrix.reinit (sp);

	//      BlockDynamicSparsityPattern dsp(n_blocks, n_blocks);
	//      for (unsigned int i=0; i<n_blocks; ++i)
	//        for (unsigned int j=0; j<n_blocks; ++j)
	//          dsp.block(i,j).reinit(dofs_per_block[i],
	//                                dofs_per_block[j]);
	//      dsp.collect_sizes();
	//
	//      DoFTools::make_sparsity_pattern (dof_handler, dsp,
	//                                       hanging_node_constraints, false);
	//      BlockSparsityPattern sparsity;
	//      sparsity.copy_from(dsp);
	//      SparsityTools::distribute_sparsity_pattern (sparsity,
	//                                                  dof_handler.n_locally_owned_dofs_per_processor(),
	//                                                  mpi_communicator,
	//                                                  locally_relevant_dofs);
	//
	//      system_matrix.reinit (//locally_owned_partitioning,
	//                            locally_owned_partitioning,
	//                            dsp,
	//                            mpi_communicator);

	system_rhs.reinit (locally_owned_partitioning,
			locally_relevant_partitioning,
			mpi_communicator,
			true);
	solution.reinit (locally_owned_partitioning,
			locally_relevant_partitioning,
			mpi_communicator,
			true);
	locally_relevant_solution.reinit (locally_relevant_partitioning,
			mpi_communicator);
	locally_relevant_solution_t1.reinit (locally_relevant_partitioning,
			mpi_communicator);


}

template<int dim>
void
CoupledProblem<dim>::make_constraints (const unsigned int newton_iteration, const unsigned int timestep)
{
	TimerOutput::Scope timer_scope (computing_timer, "Make constraints");

	if (newton_iteration >= 2)
	{
		pcout << std::string(14, ' ') << std::flush;
		return;
	}
	if (newton_iteration == 0)
	{
		dirichlet_constraints.clear();
		dirichlet_constraints.reinit (locally_relevant_dofs);

		pcout << "  CST T" << std::flush;

		const double temperature_difference_per_ts = Parameters::temperature_difference/static_cast<double>(Parameters::n_timesteps);
		if (timestep==1)
		{
			// Prescribed temperature at inner radius
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_bottom,
					ConstantFunction<dim>(293+temperature_difference_per_ts,n_components),
					dirichlet_constraints,
					fe_cell.component_mask(temperature));

			// Prescribed temperature at outer radius
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_top,
					ConstantFunction<dim>(293+temperature_difference_per_ts,n_components),
					dirichlet_constraints,
					fe_cell.component_mask(temperature));

		}
		else
		{

			// Prescribed temperature at inner radius
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_bottom,
					ConstantFunction<dim>(temperature_difference_per_ts,n_components),
					dirichlet_constraints,
					fe_cell.component_mask(temperature));

			// Prescribed temperature at outer radius
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_top,
					ConstantFunction<dim>(temperature_difference_per_ts,n_components),
					dirichlet_constraints,
					fe_cell.component_mask(temperature));
		}


		pcout << "  CST M" << std::flush;
		{
			const double axial_displacement_per_ts = Parameters::axial_displacement/static_cast<double>(Parameters::n_timesteps);
			const double radial_displacement_per_ts = Parameters::radial_displacement/(static_cast<double>(Parameters::n_timesteps));
			const double potential_difference_per_ts = Parameters::potential_difference/(static_cast<double>(Parameters::n_timesteps));

			// Bottom face
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_bottom,
					ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_cell.component_mask(z_displacement));

			// Back face
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_back,
					ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_cell.component_mask(x_displacement));

			// Left face
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_left,
					ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_cell.component_mask(y_displacement));


			// Prescribed voltage at lower surface
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_top,
					//ZeroFunction<dim>(n_components),
					ConstantFunction<dim>(+potential_difference_per_ts/2,n_components),
					dirichlet_constraints,
					fe_cell.component_mask(voltage));
			// Prescribed voltage at upper surface
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_bottom,
					//ZeroFunction<dim>(n_components),
					ConstantFunction<dim>(-potential_difference_per_ts/2,n_components),
					dirichlet_constraints,
					fe_cell.component_mask(voltage));


			// Radial displacement on internal surface
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_top,
					//ZeroFunction<dim>(n_components),
					ConstantFunction<dim>(radial_displacement_per_ts,n_components),
					dirichlet_constraints,
					fe_cell.component_mask(z_displacement));


		}



		dirichlet_constraints.close();
	}
	else
	{
		pcout << "   CST ZERO   " << std::flush;
		// Remove inhomogenaities
		for (types::global_dof_index d=0; d<dof_handler.n_dofs(); ++d)
			if (dirichlet_constraints.can_store_line(d) == true)
				if (dirichlet_constraints.is_constrained(d) == true)
					if (dirichlet_constraints.is_inhomogeneously_constrained(d) == true)
						dirichlet_constraints.set_inhomogeneity(d,0.0);
	}

	// Combine constraint matrices
	all_constraints.clear();
	all_constraints.reinit (locally_relevant_dofs);
	all_constraints.merge(hanging_node_constraints);
	all_constraints.merge(dirichlet_constraints, ConstraintMatrix::left_object_wins);
	all_constraints.close();
}


template<int dim>
void
CoupledProblem<dim>::assemble_system_thermo (const unsigned int newton_iteration, const unsigned int ts)
{
	TimerOutput::Scope timer_scope (computing_timer, "Assembly: Thermal");
	pcout << "  ASM T" << std::flush;

	FEValues<dim> fe_values(fe_cell,
			qf_cell,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);
	FEFaceValues<dim> fe_face_values(fe_cell,
			qf_face,
			update_values |
			update_normal_vectors |
			update_quadrature_points |
			update_JxW_values);

	const unsigned int dofs_per_cell = fe_cell.dofs_per_cell;
	const unsigned int n_q_points_cell = qf_cell.size();
	const unsigned int n_q_points_face = qf_face.size();

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double> cell_rhs(dofs_per_cell);
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	const bool have_heat_source = false;
	const bool have_boundary_flux = false;
	//      const ThermalNeumannBoundary<dim> neumann_boundary(have_boundary_flux);

	typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell != endc; ++cell)
	{
		if (cell->is_locally_owned() == false) continue;

		fe_values.reinit(cell);
		cell_matrix = 0;
		cell_rhs = 0;

		// Values at integration points
		std::vector< Tensor<2,dim> > Grad_u(n_q_points_cell); // Material gradient of displacement
		std::vector< Tensor<2,dim> > Grad_u_t1(n_q_points_cell); // Material gradient of displacement
		std::vector< Tensor<1,dim> > Grad_V(n_q_points_cell); // Material gradient of voltage
		std::vector< Tensor<1,dim> > Grad_V_t1(n_q_points_cell); // Material gradient of voltage
		std::vector< Tensor<1,dim> > Grad_T(n_q_points_cell); // Material gradient of temperature
		std::vector<double> theta(n_q_points_cell); // Temperature

		for (unsigned int q_point = 0; q_point < n_q_points_cell; ++q_point)
		{
			const double &JxW = fe_values.JxW(q_point);

			fe_values[displacement].get_function_gradients(locally_relevant_solution, Grad_u);
			fe_values[displacement].get_function_gradients(locally_relevant_solution_t1, Grad_u_t1);
			fe_values[voltage].get_function_gradients(locally_relevant_solution, Grad_V);
			fe_values[voltage].get_function_gradients(locally_relevant_solution_t1, Grad_V_t1);
			fe_values[temperature].get_function_gradients(locally_relevant_solution, Grad_T);
			fe_values[temperature].get_function_values(locally_relevant_solution, theta);


			//Deformation gradient at quadrature point
			const Tensor<2,dim> F_q_point = (static_cast< Tensor<2,dim> >(unit_symmetric_tensor<dim>()) + Grad_u[q_point]);
			const Tensor<2,dim> F_q_point_t1 = (static_cast< Tensor<2,dim> >(unit_symmetric_tensor<dim>()) + Grad_u_t1[q_point]);

			const double J = determinant(F_q_point);
			const double J_t1 = determinant(F_q_point_t1);
			const Tensor<1,dim> E_t1 = -(Grad_V_t1[q_point]);

			const Continuum_Point cp (F_q_point,
					-Grad_V[q_point],
					-Grad_T[q_point],
					theta[q_point]);

			// Kinematic quantites
			const Tensor<1,dim> Q = cp.get_Q();

			// Material tangents
			const Tensor<2,dim> K = cp.get_K();

			for (unsigned int i = 0; i < dofs_per_cell; ++i)
			{
				const unsigned int component_i = fe_cell.system_to_component_index(i).first;
				const unsigned int i_group     = fe_cell.system_to_base_index(i).first.first;

				const Tensor<1,dim> &Grad_Nx_i_T = fe_values[temperature].gradient(i, q_point);
				const double &Nx_i_T = fe_values[temperature].value(i, q_point);

				for (unsigned int j = 0; j < dofs_per_cell; ++j)
				{
					const unsigned int component_j = fe_cell.system_to_component_index(j).first;
					const unsigned int j_group     = fe_cell.system_to_base_index(j).first.first;

					const Tensor<1,dim> &Grad_Nx_j_T = fe_values[temperature].gradient(j, q_point);
					const double &Nx_j_T = fe_values[temperature].value(j, q_point);

					if ((i_group == T_dof) && (j_group == T_dof))
					{
						// T-T terms
						cell_matrix(i, j) -= (Grad_Nx_i_T*K*Grad_Nx_j_T) * JxW;
					}
				}

				// RHS = -Residual
				if (i_group == T_dof)
				{
					// T terms
					cell_rhs(i) -= (Grad_Nx_i_T*Q) * JxW;

				}
			}
		}


		cell->get_dof_indices(local_dof_indices);
		all_constraints.distribute_local_to_global(cell_matrix, cell_rhs,
				local_dof_indices,
				system_matrix, system_rhs);
	}

	system_matrix.compress (VectorOperation::add);
	system_rhs.compress (VectorOperation::add);
}

template<int dim>
void
CoupledProblem<dim>::assemble_system_mech (const unsigned int ts)
{
	TimerOutput::Scope timer_scope (computing_timer, "Assembly: Mechanical");
	pcout << "  ASM M" << std::flush;

	FEValues<dim> fe_values(fe_cell,
			qf_cell,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

	const unsigned int dofs_per_cell = fe_cell.dofs_per_cell;
	const unsigned int n_q_points = qf_cell.size();

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double> cell_rhs(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	// Values at integration points
	std::vector< Tensor<2,dim> > Grad_u(n_q_points); // Material gradient of displacement
	std::vector< Tensor<1,dim> > Grad_V(n_q_points); // Material gradient of voltage
	std::vector< Tensor<1,dim> > Grad_T(n_q_points); // Material gradient of temperature
	std::vector<double> theta(n_q_points); // Temperature
	unsigned int node_nr;
	typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell != endc; ++cell)
	{
		if (cell->is_locally_owned() == false) continue;

		cell_matrix = 0;
		cell_rhs = 0;
		fe_values.reinit(cell);

		fe_values[displacement].get_function_gradients(locally_relevant_solution, Grad_u);
		fe_values[voltage].get_function_gradients(locally_relevant_solution, Grad_V);
		fe_values[temperature].get_function_gradients(locally_relevant_solution, Grad_T);
		fe_values[temperature].get_function_values(locally_relevant_solution, theta);



		for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
		{
			const double &JxW = fe_values.JxW(q_point);

			const Tensor<2,dim> F_q_point = static_cast< Tensor<2,dim> >(unit_symmetric_tensor<dim>()) + Grad_u[q_point];
			const Tensor<2,dim> F_q_transpose = transpose(F_q_point);

			const Continuum_Point cp (F_q_point,
					-Grad_V[q_point],
					-Grad_T[q_point],
					theta[q_point]);

			// Kinematic quantites
			const SymmetricTensor<2,dim> S = cp.get_S();
			const Tensor<1,dim> D = cp.get_D();
			const Tensor<2,dim> S_ns (S);

			// Material tangents
			const SymmetricTensor<4,dim> C = cp.get_C();
			const Tensor<3,dim> P = cp.get_P();
			const Tensor<2,dim> DD = cp.get_DD();

			// Variation / Linearisation of Green-Lagrange strain tensor
			std::vector< SymmetricTensor<2,dim> > dE (dofs_per_cell);
			for (unsigned int k = 0; k < dofs_per_cell; ++k)
			{
				const unsigned int k_group = fe_cell.system_to_base_index(k).first.first;
				if (k_group == u_dof)
					dE[k] = symmetrize(F_q_transpose*fe_values[displacement].gradient(k, q_point));
			}

			for (unsigned int i = 0; i < dofs_per_cell; ++i)
			{
				const unsigned int component_i = fe_cell.system_to_component_index(i).first;
				const unsigned int i_group     = fe_cell.system_to_base_index(i).first.first;

				const Tensor<2,dim> &Grad_Nx_i_u      = fe_values[displacement].gradient(i, q_point);
				const Tensor<2,dim> &symm_Grad_Nx_i_u = fe_values[displacement].symmetric_gradient(i, q_point);
				const Tensor<1,dim> &Grad_Nx_i_V      = fe_values[voltage].gradient(i, q_point);

				for (unsigned int j = 0; j < dofs_per_cell; ++j)
				{
					const unsigned int component_j = fe_cell.system_to_component_index(j).first;
					const unsigned int j_group     = fe_cell.system_to_base_index(j).first.first;

					const Tensor<2,dim> &Grad_Nx_j_u      = fe_values[displacement].gradient(j, q_point);
					const Tensor<2,dim> &symm_Grad_Nx_j_u = fe_values[displacement].symmetric_gradient(j, q_point);
					const Tensor<1,dim> &Grad_Nx_j_V      = fe_values[voltage].gradient(j, q_point);

					if ((i_group == u_dof) && (j_group == u_dof))
					{
						// u-u terms: Material tangent; See Wriggers p97, p129; Holzapfel p396
						cell_matrix(i, j) += dE[i] * C * dE[j] * JxW;

						// u-u terms: Geometric tangent; See Wriggers p97, p129 eq 4.69 ; Holzapfel p396
						const SymmetricTensor<2,dim> DdE = symmetrize(transpose(Grad_Nx_j_u)*Grad_Nx_i_u);
						cell_matrix(i, j) += DdE * S * JxW;
					}
					else if ((i_group == u_dof) && (j_group == V_dof))
					{
						// u-V terms
						// Note definition of P = -2 d2_Psi/dE_dC
						cell_matrix(i, j) += contract3(Tensor<2,dim>(dE[i]), P, Grad_Nx_j_V) * JxW;

					}
					else if ((i_group == V_dof) && (j_group == u_dof))
					{
						// V-u terms
						// Note: Transpose of P not directly constructed, but we rather take into consideration
						//       the contracting DOF indices and the symmetry of those indices (P = -2 d2_Psi/dE_dC))
						 cell_matrix(i, j) += contract3(Tensor<2,dim>(dE[j]), P, Grad_Nx_i_V ) * JxW;

					}
					else if ((i_group == V_dof) && (j_group == V_dof))
					{
						// V-V terms
						cell_matrix(i, j) -= (Grad_Nx_i_V * DD * Grad_Nx_j_V) * JxW;
					}

				}

				// RHS = -Residual
				if (i_group == u_dof)
				{
					// u-terms
					cell_rhs(i) -= (dE[i]*S) * JxW;
				}
				else if (i_group == V_dof)
				{
					// V-terms
					cell_rhs(i) -= (Grad_Nx_i_V*D) * JxW;
				}
			}
		}

		cell->get_dof_indices(local_dof_indices);
		all_constraints.distribute_local_to_global(cell_matrix, cell_rhs,
				local_dof_indices,
				system_matrix, system_rhs);

	}

	system_matrix.compress (VectorOperation::add);
	system_rhs.compress (VectorOperation::add);
}

template<int dim>
void
CoupledProblem<dim>::solve_thermo (LA::MPI::BlockVector &locally_relevant_solution_update)
{
	TimerOutput::Scope timer_scope (computing_timer, "Solve: Thermal");
	pcout << "  SLV T" << std::flush;

	//      const std::string solver_type = "Iterative";
	const std::string solver_type = "Direct";

	LA::MPI::BlockVector
	completely_distributed_solution_update (locally_owned_partitioning,
			mpi_communicator);

	if (Parameters::solver_type_T == "Iterative")
	{
		SolverControl solver_control(system_matrix.block(T_block, T_block).m(),
				Parameters::tol_rel_T*system_rhs.block(T_block).l2_norm());
		LA::SolverCG solver(solver_control);
		//        LA::SolverCG solver(solver_control, mpi_communicator);

		LA::MPI::PreconditionAMG preconditioner;
		LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_TRILINOS_LA
		/* Trilinos defaults are good */
#else
		data.symmetric_operator = true;
#endif

		solver.solve(system_matrix.block(T_block, T_block),
				completely_distributed_solution_update.block(T_block),
				system_rhs.block(T_block),
				preconditioner);
	}
	else
	{ // Direct solver
#ifdef USE_TRILINOS_LA
		SolverControl solver_control(1, 1e-12);
		TrilinosWrappers::SolverDirect solver (solver_control);


		solver.solve(system_matrix.block(T_block, T_block),
				completely_distributed_solution_update.block(T_block),
				system_rhs.block(T_block));
#else
		AssertThrow(false, ExcNotImplemented());
#endif
	}

	all_constraints.distribute(completely_distributed_solution_update);
	locally_relevant_solution_update.block(T_block) = completely_distributed_solution_update.block(T_block);
}

template<int dim>
void
CoupledProblem<dim>::solve_mech (LA::MPI::BlockVector &locally_relevant_solution_update)
{
	TimerOutput::Scope timer_scope (computing_timer, "Solve: Mechanical");
	pcout << "  SLV M" << std::flush;

	LA::MPI::BlockVector
	completely_distributed_solution_update (locally_owned_partitioning,
			mpi_communicator);

	if (Parameters::solver_type_EM == "Iterative")
	{
		SolverControl solver_control(system_matrix.block(uV_block,uV_block).m(),
				Parameters::tol_rel_EM*system_rhs.block(uV_block).l2_norm());
		//        LA::SolverCG solver(solver_control, mpi_communicator);
		LA::SolverCG solver(solver_control);

		LA::MPI::PreconditionAMG preconditioner;
		LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_TRILINOS_LA
		/* Trilinos defaults are good */
#else
		data.symmetric_operator = true;
#endif

		solver.solve(system_matrix.block(uV_block, uV_block),
				completely_distributed_solution_update.block(uV_block),
				system_rhs.block(uV_block),
				preconditioner);
	}
	else
	{ // Direct solver
#ifdef USE_TRILINOS_LA
		SolverControl solver_control(1, 1e-12);
		TrilinosWrappers::SolverDirect solver (solver_control);

		solver.solve(system_matrix.block(uV_block, uV_block),
				completely_distributed_solution_update.block(uV_block),
				system_rhs.block(uV_block));
#else
		AssertThrow(false, ExcNotImplemented());
#endif
	}

	all_constraints.distribute(completely_distributed_solution_update);
	locally_relevant_solution_update.block(uV_block) = completely_distributed_solution_update.block(uV_block);

}

template<int dim>
void
CoupledProblem<dim>::output_results (const unsigned int timestep) const
{
	TimerOutput::Scope timer_scope (computing_timer, "Post-processing");

	// Write out main data file
	struct Filename
	{
		static std::string get_filename_vtu (unsigned int process,
				unsigned int cycle,
				const unsigned int n_digits = 4)
		{
			std::ostringstream filename_vtu;
			filename_vtu
			<< "solution-"
			<< (std::to_string(dim) + "d")
			<< "."
			<< Utilities::int_to_string (process, n_digits)
			<< "."
			<< Utilities::int_to_string(cycle, n_digits)
			<< ".vtu";
			return filename_vtu.str();
		}

		static std::string get_filename_pvtu (unsigned int timestep,
				const unsigned int n_digits = 4)
		{
			std::ostringstream filename_vtu;
			filename_vtu
			<< "solution-"
			<< (std::to_string(dim) + "d")
			<< "."
			<< Utilities::int_to_string(timestep, n_digits)
			<< ".pvtu";
			return filename_vtu.str();
		}

		static std::string get_filename_pvd (void)
		{
			std::ostringstream filename_vtu;
			filename_vtu
			<< "solution-"
			<< (std::to_string(dim) + "d")
			<< ".pvd";
			return filename_vtu.str();
		}
	};

	DataOut<dim> data_out;
	data_out.attach_dof_handler (dof_handler);

	std::vector<std::string> solution_names (n_components, "displacement");
	solution_names[V_component] = "voltage";
	solution_names[T_component] = "temperature";

	std::vector<std::string> residual_names (solution_names);

	for (unsigned int i=0; i < solution_names.size(); ++i)
	{
		solution_names[i].insert(0, "soln_");
		residual_names[i].insert(0, "res_");
	}

	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	data_component_interpretation(dim,
			DataComponentInterpretation::component_is_part_of_vector);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

	data_out.add_data_vector(locally_relevant_solution, solution_names,
			DataOut<dim>::type_dof_data,
			data_component_interpretation);

	LA::MPI::BlockVector locally_relevant_residual;
	locally_relevant_residual.reinit (locally_relevant_partitioning,
			mpi_communicator);
	locally_relevant_residual = system_rhs;
	locally_relevant_residual *= -1.0;
	data_out.add_data_vector(locally_relevant_residual, residual_names,
			DataOut<dim>::type_dof_data,
			data_component_interpretation);



	Vector<float> subdomain (triangulation.n_active_cells());
	for (unsigned int i=0; i<subdomain.size(); ++i)
		subdomain(i) = triangulation.locally_owned_subdomain();
	data_out.add_data_vector (subdomain, "subdomain");

	data_out.build_patches (poly_order);

	const std::string filename_vtu = Filename::get_filename_vtu(this_mpi_process,
			timestep);
	std::ofstream output(filename_vtu.c_str());
	data_out.write_vtu(output);

	// Collection of files written in parallel
	// This next set of steps should only be performed
	// by master process
	if (this_mpi_process == 0)
	{
		// List of all files written out at this timestep by all processors
		std::vector<std::string> parallel_filenames_vtu;
		for (unsigned int p=0; p < n_mpi_processes; ++p)
		{
			parallel_filenames_vtu.push_back(Filename::get_filename_vtu(p,
					timestep));
		}

		const std::string filename_pvtu (Filename::get_filename_pvtu(timestep));
		std::ofstream pvtu_master(filename_pvtu.c_str());
		data_out.write_pvtu_record(pvtu_master,
				parallel_filenames_vtu);

		// Time dependent data master file
		static std::vector<std::pair<double,std::string> > time_and_name_history;
		time_and_name_history.push_back (std::make_pair (timestep,
				filename_pvtu));
		const std::string filename_pvd (Filename::get_filename_pvd());
		std::ofstream pvd_output (filename_pvd.c_str());
		data_out.write_pvd_record (pvd_output, time_and_name_history);
	}
}

struct L2_norms
{
	L2_norms (const unsigned int block,
			const std::vector<IndexSet> &locally_owned_partitioning,
			const std::vector<IndexSet> &locally_relevant_partitioning,
			const MPI_Comm &mpi_communicator)
	: block (block),
	  locally_owned_partitioning (locally_owned_partitioning),
	  locally_relevant_partitioning (locally_relevant_partitioning),
	  mpi_communicator (mpi_communicator)
	{}

	const unsigned int block;
	const std::vector<IndexSet> &locally_owned_partitioning;
	const std::vector<IndexSet> &locally_relevant_partitioning;
	const MPI_Comm &mpi_communicator;

	double value = 1.0;
	double value_norm = 1.0;

	void
	set (const LA::MPI::BlockVector & vector,
			const ConstraintMatrix & all_constraints)
	{
		LA::MPI::BlockVector vector_zeroed;
		vector_zeroed.reinit (locally_owned_partitioning,
				locally_relevant_partitioning,
				mpi_communicator,
				true);
		vector_zeroed = vector;
		all_constraints.set_zero(vector_zeroed);

		value = vector_zeroed.block(block).l2_norm();

		// Reset if unsensible values
		if (value == 0.0) value = 1.0;
		value_norm = value;
	}

	void
	normalise (const L2_norms & norm_0)
	{
		value_norm/=norm_0.value;
	}
};

template<int dim>
void
CoupledProblem<dim>::solve_nonlinear_timestep (const double time, const int ts)
{
	L2_norms ex_T  (T_block,
			locally_owned_partitioning,
			locally_relevant_partitioning,
			mpi_communicator);
	L2_norms ex_uV (uV_block,
			locally_owned_partitioning,
			locally_relevant_partitioning,
			mpi_communicator);

	locally_relevant_solution_t1 = locally_relevant_solution;


	L2_norms res_T_0(ex_T), update_T_0(ex_T);
	L2_norms res_T(ex_T), update_T(ex_T);
	L2_norms res_uV_0(ex_uV), update_uV_0(ex_uV);
	L2_norms res_uV(ex_uV), update_uV(ex_uV);

	pcout
	<< std::string(52,' ')
	<< "|"
	<< "  RES_T  " << std::string(2,' ')
	<< "  NUP_T  " << std::string(2,' ')
	<< "  RES_UV " << std::string(2,' ')
	<< "  NUP_UV "
	<< std::endl;

	for (unsigned int n=0; n < Parameters::max_newton_iterations; ++n)
	{
		pcout << "IT " << n << std::flush;

		LA::MPI::BlockVector locally_relevant_solution_update;
		locally_relevant_solution_update.reinit (locally_relevant_partitioning,
				mpi_communicator);

		make_constraints(n, ts);

		// === THERMAL PROBLEM ===

		system_matrix = 0;
		system_rhs = 0;
		locally_relevant_solution_update = 0;

		assemble_system_thermo(n, ts);
		solve_thermo(locally_relevant_solution_update);
		locally_relevant_solution.block(T_block) += locally_relevant_solution_update.block(T_block);
		//      locally_relevant_solution.compress (VectorOperation::add);

		// Compute temperature residual
		{
			res_T.set(system_rhs, all_constraints);
			update_T.set(locally_relevant_solution_update,
					all_constraints);

			if (n == 0 || n == 1)
			{
				res_T_0.set(system_rhs, all_constraints);
				update_T_0.set(locally_relevant_solution_update,
						all_constraints);
			}

			res_T.normalise(res_T_0);
			update_T.normalise(update_T_0);
		}

		// === ELECTRO-MECHANICAL PROBLEM ===

		system_matrix = 0;
		system_rhs = 0;
		locally_relevant_solution_update = 0;

		assemble_system_mech(ts);
		solve_mech(locally_relevant_solution_update);
		locally_relevant_solution.block(uV_block) += locally_relevant_solution_update.block(uV_block);
		//      locally_relevant_solution.compress (VectorOperation::add);

		// To analyse the residual, we must reassemble both
		// systems since they depend on one another
		//      assemble_system_thermo();
		//      assemble_system_mech();

		// Compute electro-mechanical residual
		{
			res_uV.set(system_rhs, all_constraints);
			update_uV.set(locally_relevant_solution_update,
					all_constraints);

			if (n == 0 || n == 1)
			{
				res_uV_0.set(system_rhs, all_constraints);
				update_uV_0.set(locally_relevant_solution_update,
						all_constraints);
			}

			res_uV.normalise(res_uV_0);
			update_uV.normalise(update_uV_0);
		}

		pcout
		<< std::fixed
		<< std::setprecision(3)
		<< std::setw(7)
		<< std::scientific
		<< "|"
		<< "  " << res_T.value_norm
		<< "  " << update_T.value_norm
		<< "  " << res_uV.value_norm
		<< "  " << update_uV.value_norm
		<< std::endl;

		bool converged_abs=false;
		bool converged_rel=false;

		{
			if((res_T.value < Parameters::max_res_abs) &&
					(res_uV.value < Parameters::max_res_abs))
			{
				converged_abs = true;
			}

			if((res_T.value_norm < Parameters::max_res_T_norm) &&
					(res_uV.value_norm < Parameters::max_res_uV_norm))
			{
				converged_rel = true;
			}
		}

		if (converged_abs || converged_rel)
		{
			pcout
			<< "Converged."
			<< std::endl;
			break;
		}

		if (n == (Parameters::max_newton_iterations-1))
		{
			pcout
			<< "No convergence... :-/"
			<< std::endl;
		}
	}

	pcout
	<< "Absolute values of residuals and Newton update:"
	<< std::endl
	<< "res_T:  " << res_T.value
	<< "\t update_T:  " << update_T.value
	<< std::endl
	<< "res_uV: " << res_uV.value
	<< "\t update_uV: " << update_uV.value
	<< std::endl;

}


template<int dim>
void
CoupledProblem<dim>::refine_grid ()
{
	std::vector<const TrilinosWrappers::MPI::BlockVector *> storage_soln (2);
	storage_soln[0] = &locally_relevant_solution;
	storage_soln[1] = &locally_relevant_solution_t1;

	parallel::distributed::SolutionTransfer<dim,LA::MPI::BlockVector>
	soln_trans(dof_handler);

	{
		TimerOutput::Scope timer_scope (computing_timer, "Grid refinement");

		pcout
		<< "Executing grid refinement..."
		<< std::endl;

		// Estimate solution error
		Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
		KellyErrorEstimator<dim>::estimate (dof_handler,
				QGauss<dim-1>(poly_order+2),
				typename FunctionMap<dim>::type(),
				locally_relevant_solution,
				estimated_error_per_cell);

		// Perform grid marking
		parallel::distributed::GridRefinement::
		refine_and_coarsen_fixed_number (triangulation,
				estimated_error_per_cell,
				Parameters::frac_refine,
				Parameters::frac_coarsen);

		// Limit refinement level
		if (triangulation.n_levels() > Parameters::max_grid_level)
			for (typename Triangulation<dim>::active_cell_iterator
					cell = triangulation.begin_active(Parameters::max_grid_level);
					cell != triangulation.end(); ++cell)
				cell->clear_refine_flag ();

		// Prepare solution transfer for refinement
		triangulation.prepare_coarsening_and_refinement();
		soln_trans.prepare_for_coarsening_and_refinement(storage_soln);

		// Perform grid refinement
		triangulation.execute_coarsening_and_refinement ();
	}

	// Reconfigure system with new DoFs
	setup_system();

	{
		TimerOutput::Scope timer_scope (computing_timer, "Grid refinement");

		TrilinosWrappers::MPI::BlockVector distributed_solution (system_rhs);
		TrilinosWrappers::MPI::BlockVector distributed_solution_old (system_rhs);
		std::vector<TrilinosWrappers::MPI::BlockVector *> soln_tmp (2);
		soln_tmp[0] = &(distributed_solution);
		soln_tmp[1] = &(distributed_solution_old);

		// Perform solution transfer
		soln_trans.interpolate (soln_tmp);

		hanging_node_constraints.distribute(distributed_solution);
		hanging_node_constraints.distribute(distributed_solution_old);
		locally_relevant_solution     = distributed_solution;
		locally_relevant_solution_t1 = distributed_solution_old;

		pcout
		<< "Grid refinement done."
		<< std::endl;
	}
}



template<int dim>
void
CoupledProblem<dim>::run ()
{
	make_grid();
	setup_system();
	output_results(0);

	double time = Parameters::dt;
	for (unsigned int ts = 1;
			ts<=Parameters::n_timesteps;
			++ts, time += Parameters::dt)
	{
		if (Parameters::perform_AMR == true &&
				(ts % Parameters::n_ts_per_refinement) == 0)
		{
			pcout
			<< std::endl;
			refine_grid();
		}

		pcout
		<< std::endl
		<< std::string(100,'=')
		<< std::endl
		<< "Timestep: " << ts
		<< "\t Time: " << time
		<< std::endl
		<< std::string(100,'=')
		<< std::endl;
		solve_nonlinear_timestep(time, ts);
		output_results(ts);

		// Update solution at previous timestep
		locally_relevant_solution_t1 = locally_relevant_solution;
	}

}
}

int
main (int argc, char *argv[])
{
	try
	{
		using namespace dealii;
		Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
		deallog.depth_console (0);

		Coupled_TEE::CoupledProblem<3> coupled_thermo_electro_elastic_problem_3d;
		coupled_thermo_electro_elastic_problem_3d.run();
	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
		std::cerr << "Exception on processing: "
				<< std::endl
				<< exc.what()
				<< std::endl
				<< "Aborting!"
				<< std::endl
				<< "----------------------------------------------------"
				<< std::endl;

		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
		std::cerr << "Unknown exception!"
				<< std::endl
				<< "Aborting!"
				<< std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		return 1;
	}

	return 0;
}
