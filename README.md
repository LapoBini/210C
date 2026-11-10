# ECON 210C: Macroeconomics

My solutions to the problem sets and final exam of ECON 210C (graduate macroeconomics, UC San Diego,
Spring 2024, Prof. Johannes Wieland), written in Julia. The methods are explained step by step at
[lapobini.github.io/project/econ210c](https://lapobini.github.io/project/econ210c/).

## Contents

| Folder | Topic | Code |
|---|---|---|
| `HW1/` | Classical monetary model with money in the utility function, solved in sequence space | `RBC.jl` |
| `HW2/` | Monetary policy shocks: SVAR identified with a Cholesky decomposition (wild bootstrap bands) and with the Romer and Romer (2004) shock | `SVAR.jl` |
| `HW4/` | Three-equation New Keynesian model solved with undetermined coefficients: TFP and monetary policy shocks | `RANK_undetermined.jl` |
| `Exam/` | New Keynesian model with i.i.d. shocks (impulse responses, simulation, OLS vs. IV), and the effects of the Romer and Romer shock | `RANK_finalexam.jl`, `q2_code.do` |

Each folder contains the assignment (`hw*.pdf`) and the figures produced by the code (`results*/`).

## Requirements

Julia with the following packages:

```julia
using Pkg
Pkg.add(["DataFrames", "Plots", "LaTeXStrings", "CSV", "XLSX", "FredData", "HTTP", "Distributions"])
```

`HW2/SVAR.jl` and the second part of `Exam/RANK_finalexam.jl` download data from FRED, so they need a
free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html): paste it where the code says
`api_key = # INSERT YOUR FRED API KEY HERE`.

## How to run

The scripts save figures with relative paths, so start Julia from the right folder:

| Script | Run from | Figures saved in |
|---|---|---|
| `HW1/RBC.jl` | `HW1/` | `HW1/results/` |
| `HW2/SVAR.jl` | repository root | `HW2/results/` |
| `HW4/RANK_undetermined.jl` | `HW4/` | `HW4/results_TFP/` |
| `Exam/RANK_finalexam.jl` | `Exam/` | `Exam/results_Exam/` |

For example:

```bash
cd HW1
julia RBC.jl
```

`HW2/shock.xlsx` and `Exam/shock.xlsx` contain the Romer and Romer monetary policy shock
(`HW2/RR_monetary_shock_quarterly.dta` is a Stata file of the shock, not read by the code). The second part of
`Exam/RANK_finalexam.jl` writes `data.csv`, which `Exam/q2_code.do` uses to estimate the VARs in Stata;
change the `cd` path at the top of the do-file to the `Exam/` folder before running it.

## Author

Lapo Bini, UC San Diego ([lapobini.github.io](https://lapobini.github.io)).
