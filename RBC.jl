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
using DataFrames, LinearAlgebra, Dates, Statistics, Plots, GrowableArrays


# ------------------------------------------------------------------------------
# Calibration parameters
# ------------------------------------------------------------------------------
γ  = 1; 
ρ  = 1; 
χ  = 1;
β  = 0.99;
ρₘ = 0.99;
T  = 100;
v  = [0.25; 0.5; 0.999; 2; 4];
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
    aux1 = (1 - θ)^((1-γ)/ρ+γ);
    aux2 = (1 + θ^(-1/v) * (1-θ)^((1-2v)/v) * (1-β)^((v-1)/v))^(((v-γ)/(1-v))*(1/(ρ+γ)))
    aux3 = χ^(-1/(ρ+γ))

    # Compute steady state value of Cˢˢ
    Cˢˢ = aux1 * aux2 * aux3

    return Cˢˢ
    
end

function steady_state_mp(Cˢˢ, θ, β, v)

    # Compute steady state value of (M/P)ˢˢ
    MPˢˢ = (1-β)^(-1/v) * ((1-θ)/θ)^(1/v) * Cˢˢ

    return MPˢˢ

end

function steady_state_x(Cˢˢ, MPˢˢ, θ, v) 

    # Compute steady state value of Xˢˢ
    Xˢˢ = ((1-θ) * (Cˢˢ)^(1-v) + θ  * (MPˢˢ)^(1-v))^((1)/(1-v))

    return Xˢˢ

end 

# Functions to construct identity matrix, zero matrix, Differencing matrix
Iᵗ(T) = Matrix(I, T, T);
Oᵗ(T) = zeros(T, T);
Δᵗ(T) = Iᵗ(T) - [zeros(T,1) Iₜ[:,1:end-1]];


# ------------------------------------------------------------------------------
# Main Function To Solve The Model 
# ------------------------------------------------------------------------------
function RBC_solver(
    T::Int64,    # Time horizon IRF
    γ::Int64,    # Elasticity of Intertemporal substitution 
    ρ::Int64,    # Inverse Frisch elasticity of labor supply
    χ::Int64,    # Preference for leisure
    β::Float64,  # Discount Factor
    ρₘ::Float64, # Parameter AR(1) money supply shock m̂ₜ = ρₘ m̂ₜ₋₁ + ϵᵐₜ
    v::Float64,  # Parameter CES aggregator composite basket consumption vs money
    θ::Float64   # relative weights of consumption relative to money holding
    )

    # ------------------------------------------------------------------------------
    # Sequence space method works in that way: given a system of equilibrium 
    # condition of the model, I can express all of them as a function of few 
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
    #                              H(M(U,Z)) = H(U,Z) = 0
    #                              dH = dH/dU + dH/dZ = 0
    #                               dU = - (dHdU)⁻¹ dHdZ
    #
    # But since H depends on Y I have that;
    #
    #                              (i)  dHdU = ∂H/∂Y ⋅ ∂Y/∂U 
    #                              (ii) dHdZ = ∂H/∂Y ⋅ ∂Y/∂Z 
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
    ϕbdx = (γ - v) .* Δₜ;
    ϕbdp = Δₜ;
    ϕbdq = -1 .* Iₜ;

    dHdY = [ϕbdy ϕbdw ϕbdc ϕbdx ϕbdp ϕbdq];


    # ------------------------------------------------------------------------------
    # 2 - Compute ∂Y/∂U
    # ------------------------------------------------------------------------------
    C    = steady_state_c(χ, θ, β, v, γ, ρ);
    MP   = steady_state_mp(C, θ, β, v);
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
    #               Θⱼ = ∂Y/∂U ( -∂H/∂U⁻¹ ⋅ ∂H/∂Z ⋅ eⱼ ) + ∂Y/∂Z eⱼ
    #               Θⱼ = ∂Y/∂U ⋅ dU + ∂Y/∂Z eⱼ
    #
    # with:     
    #                         (i)  dHdU = ∂H/∂Y ⋅ ∂Y/∂U 
    #                         (ii) dHdZ = ∂H/∂Y ⋅ ∂Y/∂Z 
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

end






