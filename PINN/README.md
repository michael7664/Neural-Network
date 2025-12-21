This project is a small end-to-end system for turning a PDE written by a user (as LaTeX/text or as an image) into a computed solution—either an exact/analytic expression when symbolic solvers can handle it, or a Physics-Informed Neural Network (PINN) approximation when the PDE is too complex or high-dimensional for closed-form methods. It combines equation parsing, symbolic normalization, a solver “router” that selects the best backend, and a reproducible PINN training/inference pipeline that exports both numerical solution grids and physics-residual diagnostics.

## Project description
Many scientific and engineering workflows begin with a differential equation expressed in human-readable form (notes, LaTeX manuscripts, whiteboard photos), but solving that equation typically requires manual translation into a specific computational framework. This project bridges that gap by implementing a unified pipeline that (i) ingests an equation from multiple modalities, (ii) converts it into a consistent symbolic representation, and (iii) automatically attempts solution strategies ranging from classical symbolic solvers to data-free neural solvers based on the governing physics.

At a high level, the system can be viewed as a “compiler” from PDE specification to solution artifact: the input equation is parsed and normalized, problem metadata (dependent variable, independent variables, constants, domain bounds, and constraints) is loaded from a configuration file, and a routing layer decides whether to pursue an analytic or PINN-based solution. For PINN runs, the output is not only a trained neural surrogate  but also exported evaluation grids (NumPy arrays) and quantitative residual statistics that measure how well the learned solution satisfies the PDE.

# Equation ingestion and normalization
The project supports multiple equation sources:
	•	Text / LaTeX input: equations are read from  .txt / .tex  and converted into a symbolic form.
	•	Image input: equations can be extracted from an image (e.g., screenshot/scan) and then parsed into the same symbolic representation.

After ingestion, the equation is normalized into a canonical “left-hand side equals zero” residual form:\mathcal{F}(u, \partial u, \partial^2 u,\dots; x,t,\dots) = 0This normalization is crucial: it allows both symbolic solvers and PINNs to operate on a common representation of the problem, and it makes PDE residual evaluation a first-class citizen for diagnostics and training.

# Solver routing and reproducibility
A routing component decides which backend to use:
	•	Symbolic path: attempts to solve analytically (when feasible) and returns closed-form solutions or solver hints.
	•	PINN path: triggered explicitly or automatically for multi-variable PDEs and harder cases, using a                  configuration-driven setup.

A YAML configuration provides:
	•	Variable names (e.g.,  x,t )
	•	Domain bounds
	•	Physical constants (e.g., viscosity )
	•	Initial and boundary conditions (IC/BC)
	•	Training hyperparameters (steps, learning rate, sampling sizes, loss weights)
	•	Output settings (grid resolution, file paths, residual test points)
This design makes runs deterministic to reproduce (given fixed seeds) and easy to compare across hyperparameter changes—critical traits for research-style experimentation.

# PINN method and training objective
Physics-Informed Neural Networks approximate the unknown solution  with a neural network . Instead of learning from labeled solution data, the network is trained by minimizing a composite loss derived from:
	1.	PDE residual loss (physics loss) on interior collocation points
	2.	Constraint loss enforcing IC/BC targets. The project implements weighting to balance the two contributions.Derivatives in  (e.g., , , ) are computed via automatic differentiation, which allows the method to generalize across PDE forms without hand-coding derivatives for each new equation.

# Optimization strategy (Adam → LBFGS)
The training loop uses a two-stage optimizer schedule:
	•	Adam for robust initial progress in a noisy, high-dimensional loss landscape.
	•	LBFGS fine-tuning to improve convergence to a lower-loss solution once the model is in a reasonable basin.
This pattern is common in PINN practice because first-order methods provide stability early on, while quasi-Newton methods can deliver sharper convergence later, especially for smooth residual objectives.

# Outputs and diagnostics
A completed PINN run produces:
	•	A trained model 
	•	A saved solution grid (e.g.,  u_pinn.npy ) produced by evaluating the network on a mesh in the specified domain
	•	An optional visualization (e.g.,  u_pinn.png ) for 2D domains such as 
	•	Residual statistics on a large random sample of domain points:
	•	mean residual
	•	MAE
	•	RMSE
	•	max absolute residual
These diagnostics are essential for distinguishing “looks plausible” solutions from solutions that actually satisfy the PDE quantitatively.

# Example: viscous Burgers’ equation
The provided configuration demonstrates the classic viscous Burgers problem with viscosity , initial condition , and homogeneous Dirichlet boundaries. This benchmark is intentionally challenging: as  becomes small, solutions develop sharp gradients (near-shock structures), which stress both sampling and optimization—making it a useful test for validating the robustness of the PINN pipeline.

# Feature list (project capabilities)
	•	Multi-modal PDE ingestion: LaTeX/text and image-based equation input.
	•	Canonical symbolic normalization into a residual form usable by all solvers.
	•	Automatic solver selection (symbolic vs PINN) with explicit override.
	•	Config-driven problem specification (domain, constants, IC/BC, outputs).
	•	Physics-informed training with autodiff-based derivative computation.
	•	Weighted loss terms for balancing PDE residual vs constraints.
	•	Two-phase optimization: Adam pretraining + LBFGS refinement.
	•	Export of numerical solution grids ( .npy ) and optional plots ( .png ).
	•	Quantitative residual evaluation on large random test sets.