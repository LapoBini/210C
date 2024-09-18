# ------------------------------------------------------------------------------
# Solving a Benchmark New-Keynesian Model - Question 1 
# ------------------------------------------------------------------------------
# Author: Lapo Bini, lbini@ucsd.edu
# Final Exam - 06/08/2024
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# 0 - Packages
# ------------------------------------------------------------------------------
using DataFrames, LinearAlgebra, Dates, Statistics, Plots, LaTeXStrings 
using Random, Distributions
Random.seed!(1234)


# ------------------------------------------------------------------------------
# 1 - Calibration parameters
# ------------------------------------------------------------------------------
H  = 10000;
T  = 5;
σ  = 1.0; 
φ  = 1.0; 
κ  = 0.01;
β  = 0.99;
ϕₚ = 2.5;
ρᵢ = 0.0;
ρₙ = 0.0;


# ------------------------------------------------------------------------------
# 2 - Undetermined Coefficient and Simulation Function 
# ------------------------------------------------------------------------------
function IID_shock(
    T::Int64,    # Time horizon IRF
    σ::Float64,  # Elasticity of Intertemporal substitution with σ = 1/γ
    φ::Float64,  # Inverse Frisch elasticity of labor supply
    κ::Float64,  # Slope New-Keynesian Phillips curve
    β::Float64,  # Discount Factor
    ϕₚ::Float64, # Reaction parameter of Central Bank to Inflation
    ρᵢ::Float64, # Parameter AR(1) Policy shock ϵⁱₜ = ρᵢ ϵⁱₜ₋₁ + εₜⁱ
    ρₙ::Float64  # Parameter AR(1) Natural rate shock ϵⁿₜ = ρₙ ϵⁿₜ₋₁ + εₜⁿ
    )

    # ------------------------------------------------------------------------------
    # The function compute the IRF of the three equation new keynesian model to IID 
    # shocks to policy and natural rate. The IRF are computed using the method of 
    # undetermined coefficient. Benchmark three equations New-Keynesian model:
    #                      
    #                   AD : ŷₜ = Eₜ[ŷₜ₊₁] - σ( îₜ - Eₜ[πₜ₊₁] - r̂ⁿₜ )
    #                   AS : πₜ = βEₜ[πₜ₊₁] - κŷₜ       
    #                   MP : îₜ = ϕₚπₜ + īₜ    
    #
    # We are interested in two tipes of shock, to the policy and to the natural rate
    #
    #                   POL : īₜ  = εₜⁱ,  εₜⁱ ∼ N(0, σ²ᵢ)
    #                   NAT : r̂ⁿₜ = εₜⁿ,  εₜⁿ ∼ N(0, σ²ₙ)
    #
    # REMEMBER: shocks are IID, meaning that the expectation terms are zero
    # ------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
    # 1 - Unit Policy Shock 
    # ------------------------------------------------------------------------------

    # Elasticities for output and inflation 
    Ψπi = - (κ * σ)/(1 + ϕₚ * κ * σ);
    Ψŷi = - σ * (ϕₚ * Ψπi + 1);
    
    # Elasticities other variables
    Ψîi  = (ϕₚ * Ψπi) + 1;
    Ψīi  = 1;
    Ψr̂i  = Ψîi; 
    Ψr̂ⁿi = 0;

    # Dynamic System 
    Ψi = [Ψŷi Ψπi Ψîi Ψīi Ψr̂i Ψr̂ⁿi];
    εⁱ = 1 .* [(ρᵢ)^t for t in 0:T-1]; 


    # ------------------------------------------------------------------------------
    # 2 - Natural Rate Shock 
    # ------------------------------------------------------------------------------

    # Elasticities for output and inflation 
    Ψπn = (κ * σ)/(1 + ϕₚ * κ * σ);
    Ψŷn = - σ * (ϕₚ * Ψπn - 1);
    
    # Elasticities other variables
    Ψîn  = ϕₚ * Ψπn;
    Ψīn  = 0;
    Ψr̂n  = Ψîn; 
    Ψr̂ⁿn = 1;

    # Dynamic System 
    Ψn = [Ψŷn Ψπn Ψîn Ψīn Ψr̂n Ψr̂ⁿn];
    εⁿ = 1 .* [(ρₙ)^t for t in 0:T-1]; 


    # ------------------------------------------------------------------------------
    # 3 - Compute Impulse Response Functions
    # ------------------------------------------------------------------------------
    # System in steady state at t = -1 ⟶ t = 0 shock ⟶ t = 1,… system back in SS 
    IRF        = zeros(T+1, 6, 2)
    IRF[:,:,1] = [zeros(1,6); kron(Ψi, εⁱ)];
    IRF[:,:,2] = [zeros(1,6); kron(Ψn, εⁿ)];

    return IRF

end


function NK_simulation(
    H::Int64,    # Horizon - Length of the simulated time series 
    σ::Float64,  # Elasticity of Intertemporal substitution with σ = 1/γ
    φ::Float64,  # Inverse Frisch elasticity of labor supply
    κ::Float64,  # Slope New-Keynesian Phillips curve
    β::Float64,  # Discount Factor
    ϕₚ::Float64  # Reaction parameter of Central Bank to Inflation
    )

    # ------------------------------------------------------------------------------
    # Simulate the time series of the model draving εₜⁱ and εₜⁿ from their
    # Distribution 
    #                      
    #                       AD  : ŷₜ = Eₜ[ŷₜ₊₁] - σ( îₜ - Eₜ[πₜ₊₁] - r̂ⁿₜ )
    #                       AS  : πₜ = βEₜ[πₜ₊₁] - κŷₜ       
    #                       MP  : îₜ = ϕₚπₜ + īₜ    
    #                       POL : īₜ  = εₜⁱ,  εₜⁱ ∼ N(0, σ²ᵢ)
    #                       NAT : r̂ⁿₜ = εₜⁿ,  εₜⁿ ∼ N(0, σ²ₙ)
    #
    # REMEMBER: shocks are IID, meaning that the expectation terms are all zeros
    # Eₜ[πₜ₊₁] = Eₜ[ŷₜ₊₁]. Before simulating, we solve the system to obtain: 
    #
    #             AD  : ŷₜ = - (σ/1 + σκϕₚ)⋅( īₜ - r̂ⁿₜ )
    #             AS  : πₜ = - (κσ/1 + σκϕₚ)⋅( īₜ - r̂ⁿₜ )
    #             MP  : îₜ = [ 1 - (ϕₚκσ/1 + σκϕₚ)] īₜ + (ϕₚκσ/1 + σκϕₚ)r̂ⁿₜ
    #
    # ------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
    # 1 - Set up Distributions
    # ------------------------------------------------------------------------------
    # Distributions 
    Nⁱ = Normal(0, 0.001);
    Nⁿ = Normal(0, 0.01);
    
    # Random Draws
    εⁱ = rand(Nⁱ, H);
    εⁿ = rand(Nⁿ, H); 

    # ------------------------------------------------------------------------------
    # 2 - Simulate data 
    # ------------------------------------------------------------------------------
    # Coefficients for output gao and inflation 
    δ₁ = - (σ/1 + κ * σ * ϕₚ)  
    δ₂ = κ * δ₁

    # Simulate data 
    ŷₜ = δ₁ .* (εⁱ - εⁿ);
    πₜ = δ₂ .* (εⁱ - εⁿ);
    îₜ = ϕₚ .* πₜ + εⁱ; 

    # Allocate 
    SIM = [ŷₜ πₜ îₜ];

    return SIM, εⁱ

end


# ------------------------------------------------------------------------------
# 3 - Impulse Response Functions
# ------------------------------------------------------------------------------
# Allocation and prels 
var = ["Output"; "Inflation"; "Nominal Rate"; "Policy Shock"; 
       "Real Rate"; "Natural Rate Shock"];
J   = length(var);

# Create directory 
ind_dir  = readdir("./");
"results_Exam" in ind_dir ? nothing : mkdir("./results_Exam");

# Solve the model for different values of the parameters
IRF = IID_shock(T, σ, φ, κ, β, ϕₚ, ρᵢ, ρₙ);

# Generate Charts
for k in 1:J

    # Add steady state level
    plot(size =(700,400))
    hline!([0], label = "", color = "black", linewidth = 1)

    # Plot IRF
    if k == 4 
        plot!(collect(-1:1:T-1), IRF[:,k,1], title = var[k], xlabel = "Quarters",
            label = "Policy Shock", color = "purple", linewidth = 4)
    elseif k == 6
        plot!(collect(-1:1:T-1), IRF[:,k,2], title = var[k], xlabel = "Quarters",
            label = "Natural Rate Shock", color = "orange", linewidth = 4)
    else
        plot!(collect(-1:1:T-1), IRF[:,k,1], title = var[k], xlabel = "Quarters",
            label = "Policy Shock", color = "purple", linewidth = 4)
        plot!(collect(-1:1:T-1), IRF[:,k,2], title = var[k], xlabel = "Quarters",
            label = "Natural Rate Shock", color = "orange", linewidth = 4)
    end

    # Adjust plot
    plot!(xlims = (-1, T-1), xticks = collect(-1:1:T-1), 
          ytickfontsize  = 9, xtickfontsize  = 9, 
          titlefontsize = 17, yguidefontsize = 13, legendfontsize = 11, 
          boxfontsize = 15, framestyle = :box, left_margin = 4Plots.mm, 
          right_margin = 4Plots.mm, bottom_margin = 2Plots.mm, 
          top_margin = 4Plots.mm, legend = :topright)

    # Save 
    savefig("./results_Exam/"*replace(var[k], " " => "")*".pdf")
end


# ------------------------------------------------------------------------------
# 4 - Simulation 
# ------------------------------------------------------------------------------
# Allocation and prels 
var = ["Output"; "Inflation"; "Nominal Rate"];
J   = length(var);
c   = ["purple"; "orange"; "blue"];

# Simulate Model 
@time SIM, εⁱ = NK_simulation(H, σ, φ, κ, β, ϕₚ);

# Generate Charts
plot(size =(1200,800), layout = (3,1))

for j in 1:J
    plot!(collect(1:1:H), SIM[:,j], title = var[j], xlabel = "",
          color = c[j], linewidth = 1, subplot = j, label = "")
    hline!([0], label = "", color = "black", linewidth = 1, subplot = j)
end

# Adjust plot
plot!(ytickfontsize  = 9, xtickfontsize  = 9, xlims = (0,H),
      titlefontsize = 17, yguidefontsize = 13, legendfontsize = 11, 
      boxfontsize = 15, framestyle = :box, left_margin = 4Plots.mm, 
      right_margin = 4Plots.mm, bottom_margin = 2Plots.mm, 
      top_margin = 10Plots.mm, legend = :topright)

# Save 
savefig("./results_Exam/simulation.pdf")


# ------------------------------------------------------------------------------
# 5 - OLS & IV
# ------------------------------------------------------------------------------
# Compute OLS 
Y = SIM[:,1];
X = SIM[:,3];
γ = (X'*X)\(X'*Y);

x₀ = minimum(X); 
x₁ = maximum(X); 

x_aux = [collect(x₀:0.001:x₁); x₁];
y_aux = γ .* x_aux; 


plot(size =(800,800))
scatter!(SIM[:,3], SIM[:,1], alpha = 0.8, marker = :star5, markercolor = "orange", 
         label = "", markerstrokecolor = "orange",
         xlabel = "Nominal Rate", ylabel = "Output", axis = false)
plot!(x_aux, y_aux, color = "purple", lw = 4, 
      label = L"Fitted \;\; \hat y_t \;\; with \;\; \hat\gamma^{OLS} = "*string(round(γ,digits = 6)), axis = false)
hline!([0], label = "", color = "black", linewidth = 1, axis = false)
vline!([0], label = "", color = "black", linewidth = 1)
plot!(ytickfontsize  = 9, xtickfontsize  = 9,
      titlefontsize = 17, yguidefontsize = 13, legendfontsize = 11, 
      boxfontsize = 15, framestyle = :box, left_margin = 4Plots.mm, 
      right_margin = 4Plots.mm, bottom_margin = 2Plots.mm, 
      top_margin = 10Plots.mm, legend = :topright)
savefig("./results_Exam/OLS.pdf")

# Iv Estimation 
X̂     = εⁱ;
γⁱᵛ   = (X̂'*X̂)\(X̂'*Y);
z_aux = γⁱᵛ .* x_aux;

# Plot results
plot(size =(800,800))
scatter!(SIM[:,3], SIM[:,1], alpha = 0.8, marker = :star5, markercolor = "orange", 
         label = "", markerstrokecolor = "orange",
         xlabel = "Nominal Rate", ylabel = "Output", axis = false)
plot!(x_aux, y_aux, color = "purple", lw = 4, 
      label = L"Fitted \;\; \hat y_t \;\; with \;\; \hat\gamma^{OLS} = "*string(round(γ,digits = 6)), axis = false)
plot!(x_aux, z_aux, color = "blue", lw = 4, 
      label = L"Fitted \;\; \hat y_t \;\; with \;\; \hat\gamma^{IV} = "*string(round(γⁱᵛ,digits = 6)), axis = false)
hline!([0], label = "", color = "black", linewidth = 1, axis = false)
vline!([0], label = "", color = "black", linewidth = 1)
plot!(ytickfontsize  = 9, xtickfontsize  = 9,
      titlefontsize = 17, yguidefontsize = 13, legendfontsize = 11, 
      boxfontsize = 15, framestyle = :box, left_margin = 4Plots.mm, 
      right_margin = 4Plots.mm, bottom_margin = 2Plots.mm, 
      top_margin = 10Plots.mm, legend = :topright)
savefig("./results_Exam/OLS_IV.pdf")


# ------------------------------------------------------------------------------
# Romer & Romer Shock - Question 2 
# ------------------------------------------------------------------------------
# Author: Lapo Bini, lbini@ucsd.edu
# Final Exam - 06/08/2024
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# 0 - Packages
# ------------------------------------------------------------------------------
using DataFrames, Dates, CSV, XLSX, FredData, HTTP


# ------------------------------------------------------------------------------
# 1 - Download Data and Recession Dates 
# ------------------------------------------------------------------------------
# Establish connection with FRED API (Do not share API key)
api_key   = # string with your FRED API key here;
f         = Fred(api_key);

# Get monthly Data
R  = get_data(f, "FEDFUNDS", frequency = "q").data[:,end-1:end];
Y  = get_data(f, "GDPC1", frequency = "q").data[:,end-1:end];
Yᵖ = get_data(f, "GDPPOT", frequency = "q").data[:,end-1:end];
P  = get_data(f, "CPIAUCSL", frequency = "q").data[:,end-1:end];

# Load Romer & Romer Shock 
data = DataFrame(XLSX.readtable("./shock.xlsx", "Sheet2", header = true));
data = data[:,[:date, :resid_full]]

# Function to download recession dates 
function get_recessions(
    start_sample::String,
    end_sample::String;
    series        = "USRECM",
    print_csv     = false,
    download_fred = true
    )

    # ------------------------------------------------------------------------------
    # Creates a Recessions.csv file in the current directory.
    # Data are downloaded from FRED using an API
    # ------------------------------------------------------------------------------

    data = [];
    api_key   = # INSERT YOUR FRED API KEY HERE #
    f         = Fred(api_key)
    recession = get_data(f, series, frequency = "m")
    data      = recession.data;
    start     = findall(isnan.(recession.data.value))
    if isempty(start)
        data = data[:,end-1:end] |> Array{Any,2};
    else
        data = data[start[end]+1:end, end-1:end]|> Array{Any,2};
    end

    if print_csv == true
        CSV.write("./"*series*".csv",  DataFrame(data, :auto))
    end

    # Pick selected dates
    sample_bool = (data[:,1] .>= DateTime(start_sample, "dd/mm/yyyy");) .&
                  (data[:,1] .<= DateTime(end_sample, "dd/mm/yyyy"););
    data_tmp    = data[sample_bool,:];
    tmp         = findall(data_tmp[:,2].==1);
    data_tmp    = data_tmp[tmp,:];

    data_tmp[:,1] = lastdayofmonth.(data_tmp[:,1]);
    start_end     = Array{Float64,2}(undef, size(data_tmp,1), size(data_tmp,2)).*NaN;

    # Create intervals of time for the crisis
    start_end[1,1] = 1;
    for i in 1:size(data_tmp,1)-1
        idx = lastdayofmonth(data_tmp[i,1] + Month(1))
        if idx != data_tmp[i+1,1]

            srt = findall(isnan.(start_end[:,1]))[1];
            fin = findall(isnan.(start_end[:,2]))[1];

            start_end[srt,1] = i+1;
            start_end[fin,2] = i;
        end
    end

    fin              = findall(isnan.(start_end[:,2]))[1];
    start_end[fin,2] = size(data_tmp,1);
    rem              = findall(isnan.(start_end[:,2]))[1];
    idx_date         = start_end[1:rem-1,:] |> Array{Int64,2};

    # Combine to obtain tuples of interval
    rec  = [(data_tmp[idx_date[i,1],1], data_tmp[idx_date[i,2],1]) for i in 1:size(idx_date,1)];

    return rec

end

# Get recession dates
start_sample = "01/01/1970"; end_sample   = "31/10/2007"; 
rec          = get_recessions(start_sample, end_sample, series = "USRECM");


# ------------------------------------------------------------------------------
# 2 - Create Dataset 
# ------------------------------------------------------------------------------
# Create 4 quarters annualized inflation rate 
P.inf = [zeros(4); 400 .* log.(P[5:end,2]./P[1:end-4,2])];

# Create Output gap
dt_y     = innerjoin(Y, Yᵖ, on = :date, makeunique = true);
dt_y.gap = 100 .* log.(dt_y.value ./ dt_y.value_1)

# Merge everything together
df = innerjoin(data, dt_y[:,[:date, :gap]], P[:,[:date, :inf]], R, on = :date, makeunique = true);

# pi, ygap, 
rename!(df, [:date, :RR, :gap, :pi, :i])

# save Dataset
CSV.write("data.csv", df)


# ------------------------------------------------------------------------------
# 3 - Plot Data
# ------------------------------------------------------------------------------
# Names and mnemonics to save files 
var = ["Romer & Romer Shock"; "Output Gap"; "Inflation"; "Fed Funds Rate"];
sav = [L"RR_t"; L"gap_t"; L"\pi_t"; L"i_t"];

# Dates for ticks
ref_quarters = collect(Date("01/01/1970","dd/mm/yyyy"):Quarter(1):Date("01/10/2007", "dd/mm/yyyy"))[1:end];
ticks        = DateTime.(unique(year.(ref_quarters)))[1:5:end] .|> lastdayofmonth;
tck_n        = Dates.format.(Date.(ticks), "Y");

# Plot series
c = ["orange"; "purple"; "green"; "red"];
plot(layout = (4,1), size = (1200, 900))
for j in 1:4
    
    plot!(df.date .|> DateTime, df[:,j+1], color = c[j], lw = 3.5,
         label = "", title = var[j], xticks = (ticks,tck_n),
         subplot = j)
    hline!([0], color = "black", lw = 0.8, label = "", subplot = j)
    for sp in rec
        int = Dates.lastdayofmonth(sp[1].-Month(1)) |> DateTime;
        fnl = Dates.lastdayofmonth(sp[2].-Month(1)) |> DateTime;
        vspan!([int, fnl], label = "", color = "grey0",
            alpha = 0.2, subplot = j);
    end
    adj =  Dates.lastdayofmonth(rec[1][1]) |> DateTime;
    vline!([adj], label = "", alpha = 0.0, subplot =j)
end
plot!(ytickfontsize  = 10, xtickfontsize  = 10, 
        titlefontsize = 17, yguidefontsize = 13, legendfontsize = 11, 
        boxfontsize = 15, framestyle = :box, left_margin = 4Plots.mm, 
        right_margin = 4Plots.mm, bottom_margin = 2Plots.mm, 
        top_margin = 4Plots.mm, legend = :topright) 
savefig("./results_Exam/time_series.pdf")

