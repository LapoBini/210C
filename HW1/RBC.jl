# ------------------------------------------------------------------------------
# Solving a RBC Model Via Sequence space
# ------------------------------------------------------------------------------
# Author: Lapo Bini, lbini@ucsd.edu
# Problem Set 1 - Prof. Johannes Wieland - 05/15/2024
#
# Instead of using p̂ₜ as endogenous variable, I am going to use only n̂ₜ as 
# endogenous variable, which will be give me a good computational advantage. 
# The target condition that I use in H(U,Z) = 0 is the bond demand 
# equation. Enjoy
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
using DataFrames, LinearAlgebra, Dates, Statistics, Plots, LaTeXStrings


# ------------------------------------------------------------------------------
# Calibration parameters
# ------------------------------------------------------------------------------
γ  = 1.0; 
ρ  = 1.0; 
χ  = 1.0;
β  = 0.99;
ρₘ = 0.99;
T  = 500;
v  = [0.25; 0.5; 0.999; 2.0; 4.0];
θ  = zeros(5);

# Calibration Θ (relative weights consumption and money demand)
for i = 1:length(v)
    vᵤ   = v[i] ;
    θ[i] = (1 - β)/( (1-β) + (((β-1) + sqrt(β^2 - 2*β + 5))/2 )^vᵤ);
end


# ------------------------------------------------------------------------------
# Compute Steady State Values of Cˢˢ, Xˢˢ, MPˢˢ + Auxiliary Functions
# ------------------------------------------------------------------------------
function steady_state_c(χ, θ, β, v, γ, ρ)

    # Blocks
    aux1 = (1 - θ)^(1/(v-γ))
    aux2 = ((1-θ) + θ^(1/v) * ((1/((1 - θ)*(1 - β)))))^((1-v)/v)
    aux3 = χ^(-1/(v+γ)) 

    # Compute steady state value of Cˢˢ
    Cˢˢ = (aux1 * aux2 * aux3)^((v-γ)/(ρ+γ))

    return Cˢˢ
    
end;

function steady_state_mp(θ, β, v, χ, γ)

    # blocks
    aux1 = (θ/((1-θ)*(1 - β)))^(1/v);
    aux2 = χ^(-1/(v - γ))
    aux3 = (1 - θ)^(1/(v - γ))
    aux4 = ((1 - θ) + θ^(1/v) * ((1/((1-θ)*(1-β)))^((1 - v)/v)))^(1/(1-v))

    # Compute steady state value of (M/P)ˢˢ
    MPˢˢ = (aux1 * aux2 * aux3 * aux4)^((v-γ)/(ρ+γ))

    return MPˢˢ

end;

function steady_state_x(Cˢˢ, MPˢˢ, θ, v) 

    # Compute steady state value of Xˢˢ
    Xˢˢ = ((1-θ) * (Cˢˢ)^(1-v) + θ  * (MPˢˢ)^(1-v))^((1)/(1-v))

    return Xˢˢ

end;
 

# ------------------------------------------------------------------------------
# Main Function To Solve The Model 
# ------------------------------------------------------------------------------
function RBC_solver(
    T::Int64,    # Time horizon IRF
    γ::Float64,    # Elasticity of Intertemporal substitution 
    ρ::Float64,    # Inverse Frisch elasticity of labor supply
    χ::Float64,    # Preference for leisure
    β::Float64,  # Discount Factor
    ρₘ::Float64, # Parameter AR(1) money supply shock m̂ₜ = ρₘ m̂ₜ₋₁ + ϵᵐₜ
    v::Float64,  # Parameter CES aggregator composite basket consumption vs money
    θ::Float64   # relative weights of consumption relative to money holding
    )

    # ------------------------------------------------------------------------------
    # Sequence space method works in that way: given a system of equilibrium 
    # conditions of the model, I can express all of them as a function of few 
    # endogenous (U) variables and the exogenous shocks (Z):
    #
    #                                    Y = Y(U,Z)
    #
    # Then I can define a set of market clearing conditions (from my understanding 
    # they must be as much as the endogenous variables):
    #
    #                                     H(Y) = 0
    #
    # Then we have that H(Y) = H(M(U,Z)) = 0 and by taking the total derivative we get:
    #
    #                              H(Y(U,Z)) = H(U,Z) = 0
    #                              dH = (∂H/∂U)dU + (∂H/∂Z)dZ = 0
    #                              dU = - (∂H/∂U)⁻¹ (∂H/∂Z)dZ
    #
    # But since H depends on Y I have that;
    #
    #                              (i)  ∂H/∂U = ∂H/∂Y ⋅ ∂Y/∂U 
    #                              (ii) ∂H/∂Z = ∂H/∂Y ⋅ ∂Y/∂Z 
    #
    # Then I can compute the impulse response functions to the jᵗʰ exogenous shocks
    # as follow:
    #
    #                 Θⱼ = dY = ∂Y/∂U ( -∂H/∂U⁻¹ ⋅ ∂H/∂Z ⋅ eⱼ ) + ∂Y/∂Z eⱼ
    #
    # where eⱼ is the indicator vector for the structural shock. In this particular
    # case my Y is the following (coming from rearranging sligthly the log-linearized
    # equations):
    #
    # 1 - Production function:   ŷ = â + n̂
    # 2 - Real wage:             ŵ - p̂ = â
    # 3 - Market clearing:       ĉ = ŷ 
    # 4 - Labor Supply:          x̂ = (v - γ)⁻¹ [ ρn̂ + vĉ - â]
    # 5 - Composite Basket:      p̂ = m̂ + A⁻¹B ĉ - A⁻¹(v - γ) x̂
    # 6 - Money Demand:          q̂ = [v (1-β) β⁻¹] (ĉ - m̂ + p̂) 
    #
    # These are not the original log-linearised equations, but they characterize the
    # structure of my DAG. Starting from one, given the endogenous variable and the 
    # two exogenous I can characterize all the others. 
    # ------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------
    # 0 - Preliminary Stuff 
    # ------------------------------------------------------------------------------
    # Functions to construct identity matrix, zero matrix, Differencing matrix
    Iᵗ(T) = Matrix(I, T, T);
    Oᵗ(T) = zeros(T, T);
    Δᵗ(T) = -Iᵗ(T) + [zeros(T,1) Iᵗ(T)[:,1:end-1]];

    # Create identity matrix, zero matrix, and difference matrix 
    Iₜ = Iᵗ(T);
    Oₜ = Oᵗ(T);
    Δₜ = Δᵗ(T);


    # ------------------------------------------------------------------------------
    # 1 - Compute ∂H/∂Y
    # ------------------------------------------------------------------------------
    # Construct ∂H/∂Y where Hₜ is th bond demand equation in my DAG:
    #
    #             v(ĉₜ₊₁ - ĉₜ) - q̂ₜ  + (p̂ₜ₊₁ - p̂ₜ) - (v - γ)(x̂ₜ₊₁ - x̂ₜ) = 0
    #
    # Which is the money demand condition. Notice that ϕbdw is the coefficient for the 
    # real wage. My endogenous unkiwn variable is U = {n̂ₜ} while the shocks are two,
    # Z = {âₜ ; m̂ₜ} with m̂ₜ = ρₘ m̂ₜ₋₁ + ϵₜ.
    # ------------------------------------------------------------------------------
    ϕbdy = Oₜ;
    ϕbdw = Oₜ;
    ϕbdc = v .* Δₜ;
    ϕbdx = -(v-γ) .* Δₜ;
    ϕbdp = Δₜ;
    ϕbdq = -1 .* Iₜ;

    dHdY = [ϕbdy ϕbdw ϕbdc ϕbdx ϕbdp ϕbdq];


    # ------------------------------------------------------------------------------
    # 2 - Compute ∂Y/∂U
    # ------------------------------------------------------------------------------
    C    = steady_state_c(χ, θ, β, v, γ, ρ);
    MP   = steady_state_mp(θ, β, v, χ, γ);
    X    = steady_state_x(C, MP, θ, v);

    A    = θ * (MP/X)^(1-v);
    B    = (1-θ) * (C/X)^(1-v);

    ϕyn  = Iₜ;
    ϕwn  = Oₜ;
    ϕcn  = ϕyn;
    ϕxn  = ((v+ρ)/(v-γ)) .* Iₜ;
    ϕpn  = (B/A) .* ϕcn - (1/A) .* ϕxn;
    ϕqn  = ((v*(1-β))/β) .* (ϕcn + ϕpn);

    dYdU = [ϕyn; ϕwn; ϕcn; ϕxn; ϕpn; ϕqn;];


    # ------------------------------------------------------------------------------
    # 2 - Compute ∂Y/∂Z
    # ------------------------------------------------------------------------------
    # Elasticities to Total Factor Productivity Shock
    ϕya  = Iₜ;
    ϕwa  = Iₜ;
    ϕca  = ϕya;
    ϕxa  = ((v-1)/(v-γ)) .* Iₜ;
    ϕpa  = (B/A) .* ϕca - (1/A) .* ϕxa;
    ϕqa  = ((v*(1-β))/β) .* (ϕca + ϕpa);

    # Elasticity to money supply 
    ϕym  = Oₜ;
    ϕwm  = Oₜ;
    ϕcm  = Oₜ;
    ϕxm  = Oₜ;
    ϕpm  = Iₜ;
    ϕqm  = Oₜ;

    dYdZ = [ϕya ϕym; ϕwa ϕwm; ϕca ϕcm; ϕxa ϕxm; ϕpa ϕpm; ϕqa ϕqm];


    # ------------------------------------------------------------------------------
    # 3 - Compute IRF 
    # ------------------------------------------------------------------------------
    # Remember that:
    #
    #               Θⱼ = ∂Y/∂U ( -(∂H/∂U)⁻¹ ⋅ (∂H/∂Z) ⋅ eⱼ ) + ∂Y/∂Z eⱼ
    #               Θⱼ = ∂Y/∂U ⋅ dU + ∂Y/∂Z eⱼ
    #
    # with dZ = eⱼ and with:     
    #                         (i)  ∂H/∂U = ∂H/∂Y ⋅ ∂Y/∂U 
    #                         (ii) ∂H/∂Z = ∂H/∂Y ⋅ ∂Y/∂Z 
    # ------------------------------------------------------------------------------
    # Compute single blocks 
    dHdU = dHdY * dYdU;
    dHdZ = (dHdY * dYdZ);
    dUdZ = -1 .* ( dHdU \ dHdZ);

    # Know compute the IRF matrix, dimension is (nT x Tk) where k is the number of 
    # exogenous shock
    dY = dYdU * dUdZ + dYdZ;

    # Construct path for the shock. Remember that m̂ follows an AR(1) process
    ϵₐ = zeros(T,1);
    ϵₘ = 1 .* [(ρₘ)^t for t in 0:T-1]; 

    # Compute IRF
    IRF = dY * [ϵₐ; ϵₘ];
    IRF = reshape(IRF, T, 6);

    return IRF

end;


# ------------------------------------------------------------------------------
# Produce Results
# ------------------------------------------------------------------------------
# Loop over different parameters calibration 
# Allocation and prels 
T   = 500;
J   = length(v);
IRF = zeros(T, 6, J);
var = ["Output"; "Real Wage"; "Consumption"; "Composite Basket"; "Price"; "Nominal Rate"];
sav = ["output"; "real_wage"; "consumption"; "composite Basket"; "price"; "nominal_rate"]

# Create directory 
ind_dir  = readdir("./");
"results" in ind_dir ? nothing : mkdir("./results");

# Solve the model for different values of the parameters
@time for j in 1:J
    IRF[:,:,j] = RBC_solver(T,γ,ρ,χ,β,ρₘ,v[j],θ[j])
end

# Generate Charts
c  = ["Orange"; "Purple"; "Brown"; "Deepskyblue"; "Lime"];
vv = [".25"; ".5"; "1"; "2"; "4"];
for k in 1:6

    # Add steady state level
    plot(size =(700,400))
    hline!([0], label = "", color = "black", linewidth = 1)

    # Plot over different number of parameters for each variable
    for j in 1:J
        plot!(collect(1:1:T), IRF[:,k,j], title = var[k], xlabel = "Horizon",
            label = L"v = "*vv[j]*"   "*L" \theta = "*string(round(θ[j], digits = 5)),
            color = c[j], linewidth = 4)
    end

    # Adjust plot
    plot!(xlims = (0, T), xticks = collect(0:100:T), 
          ytickfontsize  = 9, xtickfontsize  = 9, 
          titlefontsize = 17, yguidefontsize = 13, legendfontsize = 9, 
          boxfontsize = 15, framestyle = :box, left_margin = 4Plots.mm, 
          right_margin = 4Plots.mm, bottom_margin = 2Plots.mm, 
          top_margin = 4Plots.mm, legend = :bottomright)

    # Save 
    savefig("./results/"*sav[k]*".pdf")
end