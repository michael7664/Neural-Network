1. Overview
This tool solves partial differential equations (PDEs) starting from a human‑readable equation (text, LaTeX, or image).
It can:
	•	Try to obtain an analytic solution via a symbolic solver.
	•	Train a Physics‑Informed Neural Network (PINN) to approximate the solution on a chosen domain.
	•	Export solution grids ( .npy ), plots ( .png ), and residual statistics.
All problem details (domain, constants, initial/boundary conditions, and training hyperparameters) are specified in a YAML configuration file. 
2. Installation
	1.	"Clone the repository:
        git clone https://github.com/michael7664/Neural-Network.git"
        "cd PINN"

    2.	Create and activate a Python environment (example with  conda ):
        "conda create -n pinn-env python=3.11"
        "conda activate pinn-env"

    3.	Install dependencies:
        "pip install -r requirements.txt"
        (Ensure PyTorch is installed with a backend that matches your machine: CPU, CUDA, or Apple MPS.)

3. Command‑line interface
The main entry point is the module  src.cli_file :
"python -m src.cli_file --input <FILE> --input-type <TYPE> --func <NAME> [OPTIONS]"

Required arguments
	•	 --input <FILE>  Path to the equation file:
	•	 .tex  or  .txt  for LaTeX / text input
	•	 .png  /  .jpg  for equation images
	•	 --input-type <TYPE>  One of:
	•	 txt-sympy  – text file containing a SymPy‑style expression
	•	 txt-latex  – LaTeX file containing the equation
	•	 image  – image file with a rendered equation
	•	 --func <NAME>  Name of the unknown function in the PDE (e.g.  u ).

Solver selection
	•	 --prefer symbolic  Force the symbolic solver. If a closed‑form solution is found, it is printed and the program exits.
	•	 --prefer pinn  Force the PINN solver. Requires a YAML configuration file.
	•	 --prefer auto  (default) Try symbolic first; if the equation is too complex or higher‑dimensional, fall back to the PINN solver.
Configuration
	•	 --config <PATH>  Path to a YAML file describing:
	•	variables and domain bounds,
	•	constants,
	•	initial/boundary conditions,
	•	PINN hyperparameters,
	•	output settings.
    For PINN runs,  --config  is mandatory.

4. Example: solving viscous Burgers’ equation
Assume:
	•	The PDE in LaTeX is stored in  eq.tex .
	•	The YAML configuration is in  config.yaml .
Run:
python -m src.cli_file \
  --input eq.tex \
  --input-type txt-latex \
  --func u \
  --prefer pinn \
  --config config.yaml
On success, the CLI prints:
	•	Which method was used:  Method: symbolic  or  Method: pinn .
	•	Optional symbolic hints or the PINN residual statistics.
	•	Paths of saved outputs (e.g.  u_pinn.npy ,  u_pinn.png ).

5. YAML configuration format
A minimal example configuration for Burgers’ equation:
vars: ["x", "t"]

domain:
  bounds:
    x: [-1.0, 1.0]
    t: [0.0, 1.0]

constants:
  nu: 0.01

conditions:
  ic:
    - sampler: {type: initial, var: t, value: 0.0}
      target:  {type: expr, expr: "-sin(pi*x)"}

  bc:
    - sampler: {type: boundary, var: x, value: -1.0}
      target:  {type: zeros}
    - sampler: {type: boundary, var: x, value:  1.0}
      target:  {type: zeros}

pinn:
  steps: 10000        # Adam steps
  lr: 0.001
  n_f: 20000          # interior collocation points
  cond_batch: 256
  print_every: 200
  w_f: 1.0            # weight for PDE residual
  w_c: 10.0           # weight for IC/BC
  lbfgs_steps: 500    # extra LBFGS iterations
  lbfgs_lr: 1.0

output:
  residual_points: 20000
  grid:
    x: 201
    t: 101
    npy: "u_pinn.npy"
    png: "u_pinn.png"

Key sections
	•	 vars  Ordered list of independent variables. For PINNs, this must match the equation (e.g.  x,t ).
	•	 domain.bounds  Closed interval for each variable, used for sampling and grid evaluation.
	•	 constants  Physical parameters such as  nu  (viscosity). These are substituted into the PDE before training.
	•	 conditions.ic  /  conditions.bc  Lists of conditions. Each item has:
	•	 sampler : describes where to sample (initial line or boundaries).
	•	 target : describes the target value:
	•	 type: expr  with an expression in  x  /  t , or
	•	 type: zeros  for homogeneous conditions.
	•	 pinn  Training hyperparameters:
	•	 steps ,  lr  – Adam optimizer settings.
	•	 n_f  – number of interior collocation points.
	•	 w_f ,  w_c  – loss weights.
	•	 lbfgs_steps ,  lbfgs_lr  – optional L‑BFGS refinement.
    •	 output 
	•	 residual_points  – number of random points used to estimate residual statistics.
	•	 grid  – resolution and filenames for saved solution grids and plots.

6. Interpreting outputs
For a PINN run, you will typically see console output like:
LBFGS done. final_loss=7.552e-04
Method: pinn
Kind: unknown
Hints: []
Saved u_pinn.npy
Saved u_pinn.png
Residual stats: {'residual_mean': ..., 'residual_mae': ..., 'residual_rmse': ..., 'residual_maxabs': ...}

LBFGS done. final_loss=7.552e-04
Method: pinn
Kind: unknown
Hints: []
Saved u_pinn.npy
Saved u_pinn.png
Residual stats: {'residual_mean': ..., 'residual_mae': ..., 'residual_rmse': ..., 'residual_maxabs': ...}

•	 u_pinn.npy  NumPy array of shape  (Nt, Nx)  (or higher‑D) containing the solution on the evaluation grid.
	•	 u_pinn.png  Heatmap plot of  for 2D problems.
	•	 Residual stats  Quantitative check of how well the network satisfies the PDE:
	•	lower  residual_rmse  and  residual_maxabs  indicate better physics satisfaction. 

7. Common workflows
1) Symbolic only

python -m src.cli_file \
  --input eq.tex \
  --input-type txt-latex \
  --func u \
  --prefer symbolic
If a closed‑form solution exists, it is printed to the terminal.

2) PINN with different hyperparameters
Edit the  pinn:  section in  config.yaml  (e.g., change  steps ,  n_f , or  w_c ), then rerun the same command.
This makes it easy to perform controlled ablations or sensitivity studies. 
3) Different PDE
	•	Put the new PDE in a  .tex  /  .txt  file.
	•	Update:
	•	 vars 
	•	 domain.bounds 
	•	 constants 
	•	 conditions 
	•	Optionally adjust  pinn:  hyperparameters.
	•	Run the CLI with the new inputs.

8. Troubleshooting
	•	File not found Ensure the paths for  --input  and  --config  are correct (e.g.  config.yaml , not  config.yam ).
	•	Dimension mismatch errors Check that  vars  in the YAML matches the variables used in the equation and in expressions like  "-sin(pi*x)" .
	•	Poor solution quality (noisy plots / high residuals) Try:
	•	increasing  steps  and/or  n_f ,
    •	raising  w_c  to better enforce boundary/initial conditions,
	•	increasing  lbfgs_steps ,
	•	or running on a different device (CPU vs GPU/MPS) if numerical issues are suspected. 




