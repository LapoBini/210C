# ------------------------------------------------------------------------------
# Solving a Benchmark New-Keynesian Model 
# ------------------------------------------------------------------------------
# Author: Lapo Bini, lbini@ucsd.edu
# Problem Set 4 - Prof. Johannes Wieland - 06/05/2024
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
using DataFrames, LinearAlgebra, Dates, Statistics, Plots, LaTeXStrings


# ------------------------------------------------------------------------------
# Calibration parameters
# ------------------------------------------------------------------------------
T  = 20;
σ  = 1.0; 
φ  = 1.0; 
κ  = 0.1;
β  = 0.99;
ϕₚ = 1.5;
ρₐ = 0.8;
ρᵥ = 0.9;


# ------------------------------------------------------------------------------
# Undetermined Coefficient 
# ------------------------------------------------------------------------------
# TFP Shock 
function TFP_shock(
    T::Int64,    # Time horizon IRF
    σ::Float64,  # Elasticity of Intertemporal substitution with σ = 1/γ
    φ::Float64,  # Inverse Frisch elasticity of labor supply
    κ::Float64,  # Slope New-Keynesian Phillips curve
    β::Float64,  # Discount Factor
    ϕₚ::Float64, # Reaction parameter of Central Bank to Inflation
    ρₐ::Float64, # Parameter AR(1) TFP shock âₜ = ρₐ âₜ₊₁ + εₜᵃ
    )

    # ------------------------------------------------------------------------------
    # The function compute the IRF of the three equation new keynesian model to a 
    # transitory TFP shock, ignoring the monetary policy shock. The IRF are computed
    # using the method of undetermined coefficient. 
    # Benchmark three equations New-Keynesian model made by:
    #                      
    #                            AD : ŷₜ = Eₜ[ŷₜ₊₁] - σ r̂ₜ
    #                            AS : π = β Eₜ [πₜ₊₁] - κ ( ŷₜ - ŷₜᶠ )       
    #                            MP : î = ϕₚ πₜ + vₜ     
    #
    # We are interested in the TFP shock, then we are going to model productivity as an 
    # AR(1): âₜ = ρₐ âₜ₊₁ + εₜᵃ while we set the MP shock always equal to zero: vₜ = 0.
    #
    # The METHOD OF UNDETERMINED COEFFICIENT could be described by 5 different steps: 
    #
    # 1 - GUESS 
    #     In our case we are going to guess a lineary policy function for log deviation
    #     of output, inflation, and their relative expectations:
    #     ŷₜ = ψⱼₐ âₜ
    #     πₜ = ψₚₐ âₜ  
    #     Eₜ[ŷₜ₊₁] = ψⱼₐ Eₜ[âₜ₊₁] = ψⱼₐ ρₐ âₜ
    #     Eₜ[πₜ₊₁] = ψₚₐ Eₜ[âₜ₊₁] = ψₚₐ ρₐ âₜ 
    #
    # 2 - INSERT
    #     Substitute the four policy equation inside the three equation of the NK model,
    #     moreover, plug monetary policy rule inside r̂ₜ since we have that the log 
    #     deviation of the real rate is: r̂ₜ = î - Eₜ[πₜ₊₁]
    #
    # 3 - COLLECT 
    #     The goal is to solve for ψⱼₐ and ψₚₐ in terms of all the exogenous parameters. 
    #
    # 4 - CALIBRATE
    #     In this case we are going to use the parameters imputed in the argument of 
    #     the function, we don't actually need to estimate anything. 
    #
    # 5 - IMPULSE RESPONSE FUNCTIONS 
    #     We express the vector of variables as a function of the exogenous shock and we
    #     compute the IRF recursively using the law of motion for TFP shock. 
    # ------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------
    # Construct parameters
    # ------------------------------------------------------------------------------
    γ  = 1/σ;
    Ω  = (1 + φ)/(φ + γ); 
    Λₐ = 1/((1 - ρₐ) * (1 - β * ρₐ) + σ * κ * (ϕₚ - ρₐ));

    # Elasticities for output and inflation 
    Ψŷa = Λₐ * Ω * σ * κ * (ϕₚ - ρₐ);
    Ψπa = (κ^2/(1-β*ρₐ)) * Λₐ * Ω * σ * (ϕₚ - ρₐ) - (κ/(1-β*ρₐ)) * Ω;
    
    Ψỹa  = Ψŷa - Ω;
    Ψŷᶠa = Ω;
    Ψîa  = ϕₚ * Ψπa;
    Ψr̂a  = ϕₚ * Ψπa - Ψπa * ρₐ;
    Ψn̂a  = Ψŷa - 1;
    Ψâa  = 1;


    # ------------------------------------------------------------------------------
    # Create Dynamic System as State Space Model
    # ------------------------------------------------------------------------------
    Ψ  = [Ψŷa Ψπa Ψỹa Ψŷᶠa Ψîa Ψr̂a Ψn̂a Ψâa];
    εᵃ = 1 .* [(ρₐ)^t for t in 0:T-1]; 


    # ------------------------------------------------------------------------------
    # Compute Impulse Response Functions
    # ------------------------------------------------------------------------------
    IRF = kron(Ψ, εᵃ);

    return IRF

end

# Monetary Policy Shock
function MP_shock(
    T::Int64,    # Time horizon IRF
    σ::Float64,  # Elasticity of Intertemporal substitution with σ = 1/γ
    φ::Float64,  # Inverse Frisch elasticity of labor supply
    κ::Float64,  # Slope New-Keynesian Phillips curve
    β::Float64,  # Discount Factor
    ϕₚ::Float64, # Reaction parameter of Central Bank to Inflation
    ρᵥ::Float64, # Parameter AR(1) MP shock vₜ = ρᵥ vₜ₋₁ + εₜᵛₜ
    )

    # ------------------------------------------------------------------------------
    # The function compute the IRF of the three equation new keynesian model to a 
    # transitory TFP shock, ignoring the monetary policy shock. The IRF are computed
    # using the method of undetermined coefficient. 
    # Benchmark three equations New-Keynesian model made by:
    #                      
    #                            AD : ŷₜ = Eₜ[ŷₜ₊₁] - σ r̂ₜ
    #                            AS : π = β Eₜ[πₜ₊₁] - κ ( ŷₜ - ŷₜᶠ )       
    #                            MP : î = ϕₚ πₜ + vₜ     
    #
    # We are interested in the MP shock, then we are going to model MP shocks as an 
    # AR(1): vₜ = ρᵥ vₜ₋₁ + εₜᵛ while we set the MP shock always equal to zero: âₜ = 0.
    #
    # The METHOD OF UNDETERMINED COEFFICIENT could be described by 5 different steps: 
    #
    # 1 - GUESS 
    #     In our case we are going to guess a lineary policy function for log deviation
    #     of output, inflation, and their relative expectations:
    #     ŷₜ = ψⱼᵥ vₜ
    #     πₜ = ψₚᵥ vₜ  
    #     Eₜ[ŷₜ₊₁] = ψⱼᵥ Eₜ[vₜ₊₁] = ψⱼᵥ ρᵥ vₜ
    #     Eₜ[πₜ₊₁] = ψₚᵥ Eₜ[vₜ₊₁] = ψₚₐ ρᵥ vₜ 
    #
    # 2 - INSERT
    #     Substitute the four policy equation inside the three equation of the NK model,
    #     moreover, plug monetary policy rule inside r̂ₜ since we have that the log 
    #     deviation of the real rate is: r̂ₜ = î - Eₜ[πₜ₊₁]
    #
    # 3 - COLLECT 
    #     The goal is to solve for ψⱼᵥ and ψₚᵥ in terms of all the exogenous parameters. 
    #
    # 4 - CALIBRATE
    #     In this case we are going to use the parameters imputed in the argument of 
    #     the function, we don't actually need to estimate anything. 
    #
    # 5 - IMPULSE RESPONSE FUNCTIONS 
    #     We express the vector of variables as a function of the exogenous shock and we
    #     compute the IRF recursively using the law of motion for MP shock. 
    # ------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------
    # Construct parameters
    # ------------------------------------------------------------------------------
    γ  = 1/σ;
    Λᵥ = 1/(1 + ((σ * κ)/(1 - β * ρᵥ)) * (ϕₚ - ρᵥ) - ρᵥ);

    # Elasticities for output and inflation 
    Ψŷv = -Λᵥ * σ;
    Ψπv = (κ/(1-β*ρᵥ)) * Ψŷv;
    
    Ψỹv  = Ψŷv;
    Ψŷᶠv = 0;
    Ψîv  = ϕₚ * Ψπv + 1;
    Ψr̂v  = Ψîv - Ψπv * ρᵥ;
    Ψn̂v  = Ψŷv;
    Ψvv  = 1;


    # ------------------------------------------------------------------------------
    # Create Dynamic System as State Space Model
    # ------------------------------------------------------------------------------
    Ψ  = [Ψŷv Ψπv Ψỹv Ψŷᶠv Ψîv Ψr̂v Ψn̂v Ψvv];
    εᵛ = 1 .* [(ρᵥ)^t for t in 0:T-1]; 


    # ------------------------------------------------------------------------------
    # Compute Impulse Response Functions
    # ------------------------------------------------------------------------------
    IRF = kron(Ψ, εᵛ);

    return IRF

end


# ------------------------------------------------------------------------------
# Produce Results
# ------------------------------------------------------------------------------
# Loop over different parameters calibration 
# Allocation and prels 
var = ["Output"; "Inflation"; "Output Gap"; "Potential Output"; 
       "Nominal Rate"; "Real Rate"; "Employment"; "Technology"];
J   = length(var);

# Create directory 
ind_dir  = readdir("./");
"results" in ind_dir ? nothing : mkdir("./results_TFP");

# Solve the model for different values of the parameters
@time IRFa = TFP_shock(T, σ, φ, κ, β, ϕₚ, ρₐ);
@time IRFm = MP_shock(T, σ, φ, κ, β, ϕₚ, ρᵥ);

# Generate Charts
for k in 1:J

    # Add steady state level
    plot(size =(700,400))
    hline!([0], label = "", color = "black", linewidth = 1)

    # Plot IRF
    plot!(collect(1:1:T), IRFa[:,k], title = var[k], xlabel = "Quarters",
            label = "", color = "purple", linewidth = 4)

    # Adjust plot
    plot!(xlims = (0, T), xticks = collect(0:2:T), 
          ytickfontsize  = 9, xtickfontsize  = 9, 
          titlefontsize = 17, yguidefontsize = 13, legendfontsize = 9, 
          boxfontsize = 15, framestyle = :box, left_margin = 4Plots.mm, 
          right_margin = 4Plots.mm, bottom_margin = 2Plots.mm, 
          top_margin = 4Plots.mm, legend = :bottomright)

    # Save 
    savefig("./results_TFP/"*replace(var[k], " " => "")*".pdf")
end



