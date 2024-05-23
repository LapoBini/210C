# ------------------------------------------------------------------------------
# SVAR
# ------------------------------------------------------------------------------
# Author: Lapo Bini, lbini@ucsd.edu
# Problem Set 2 - Prof. Johannes Wieland - 22/15/2024
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# 0 - Packages
# ------------------------------------------------------------------------------
using DataFrames, LinearAlgebra, Dates, Statistics, Plots, LaTeXStrings
using CSV, XLSX, FredData, HTTP


# ------------------------------------------------------------------------------
# 1 - Download FRED Data
# ------------------------------------------------------------------------------
# Establish connection with FRED API (Do not share API key)
api_key   = "66c080f0ed7880e7df1230ef212fb8c1";
f         = Fred(api_key);

# Get monthly Data
R = get_data(f, "FEDFUNDS", frequency = "m").data[:,end-1:end];
u = get_data(f, "UNRATE", frequency = "m").data[:,end-1:end];
P = get_data(f, "GDPDEF", frequency = "q").data[:,end-1:end];

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
    #
    # Author: Lapo Bini, lbini@ucsd.edu
    # ------------------------------------------------------------------------------

    data = [];
    api_key   = "66c080f0ed7880e7df1230ef212fb8c1"
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
start_sample = "01/01/1960"; end_sample   = "31/12/2024"; 
rec          = get_recessions(start_sample, end_sample, series = "USRECM");


# ------------------------------------------------------------------------------
# 2 - Plot Data
# ------------------------------------------------------------------------------
# Names and mnemonics to save files 
var = ["Fed Funds Rate"; "Unemployment Rate"; "GDP Deflator"];
sav = ["FFR"; "UNEM"; "GDPDEF"];
key = ["R", "u", "P"];
dic = Dict("R" => R, "u" => u, "P" => P)

# Create directory 
ind_dir  = readdir("./HW2");
"results" in ind_dir ? nothing : mkdir("./HW2/results");


for i in 1:length(sav)
    ref_months = dic[key[i]].date[1]:Year(5):dic[key[i]].date[end] |> collect .|> DateTime;
    ticks      = DateTime.(unique(year.(Date.(dic[key[i]].date))))[1:5:end] .|> lastdayofmonth;
    tck_n      = Dates.format.(Date.(ticks), "Y");

    plot(dic[key[i]].date .|> DateTime, dic[key[i]].value, color = "purple", lw = 2.5, label = "",
        size =(800,300), xticks = (ticks,tck_n))
    for sp in rec
        int = Dates.lastdayofmonth(sp[1].-Month(1)) |> DateTime;
        fnl = Dates.lastdayofmonth(sp[2].-Month(1)) |> DateTime;
        vspan!([int, fnl], label = "", color = "grey0",
            alpha = 0.2);
    end
    adj =  Dates.lastdayofmonth(rec[1][1]) |> DateTime;
    vline!([adj], label = "", alpha = 0.0)
    plot!(ytickfontsize  = 10, xtickfontsize  = 10, title = var[i],
        titlefontsize = 17, yguidefontsize = 13, legendfontsize = 11, 
        boxfontsize = 15, framestyle = :box, left_margin = 4Plots.mm, 
        right_margin = 4Plots.mm, bottom_margin = 2Plots.mm, 
        top_margin = 4Plots.mm, legend = :topleft)  # remove surrounding black box
    savefig("./HW2/results/"*sav[i]*".pdf")
end


# ------------------------------------------------------------------------------
# 3 - Choleski Functions
# ------------------------------------------------------------------------------
function IRF_CH(
    y,  # T x K
    p,  # lag order of the var
    H,  # horizon IRF
    )

    # --------------------------------------------------------------------------
    # IRF_CH Impulse response function through cholesky decomposition.
    # Author: Lapo Bini, lbini@ucsd.edu
    # --------------------------------------------------------------------------

    T, K = size(y);
    Y    = y[p+1:T,:]';  # 1st column = 1st obs for each K variables
                         # Y size = K x (T-(p+1))
    _, t = size(Y);
    X    = ones(1,T-p);
    
    for j=1:p
        x = y[p+1-j:T-j,:]';  # Y_t-p such that 1st column = 1st obs x each variab
        X = [X; x];           # final size: X = (1+(Kxp)) x (T-(p+1)) 
    end
    
    # OLS estimation
    B  = (Y*X')/(X*X');      # size = k x (1+(kxp)) 
    U  = Y-B*X;              # Matrix of residuals, size = K x (1+(kxp)) 
    S  = (U*U')/(t);         # Variance/Covariance Matrix while U'*U RSS
    B₀ = cholesky(S).L;
    
    # Create companion matrix and selection matrix
    A   = [B[:,2:end]; Matrix(I, K*(p-1), K*(p-1)) zeros(K*(p-1), K)];  
    J   = [Matrix(I, K, K) zeros(K, K*(p-1))];

    # Allocation results
    IRF = zeros(H+1, K, K);

    for k in 1:K 
        IRF[1,:,k] = B₀[:,k]';

        # Compute IRF
        for h=1:H
            M   = J*A^h*J';
            MM  = B₀*M;
            IRF[h+1,:,k] = MM[:,k]';
        end
    end
    
    return IRF, B, U, B₀

end

function WILD_CH(
    y,   # data
    B,   # OLS coeff
    u,   # residual
    p,   # lag order
    H,   # forecast horizon
    nrep # bootstrap repetitions
    )

    # --------------------------------------------------------------------------
    # WILD_CH Compute the confidence interval using wild bootstrap
    # It computes the confidence interval accounting for the
    # heteroskedasticity in the error term.
    #
    # How I compute the quantiles: I am going to apply the quantile function 
    # to slices of IRFS along the fourth dimension (bootstrap repetitions).
    # The function u -> quantile(u, a) computes the specified quantiles for
    # each slice. The dims=(4,) argument specifies that the function should 
    # be applied along the fourth dimension. The last dimension of the output
    # matrix will give the quantiles 
    #
    # Author: Lapo Bini, lbini@ucsd.edu
    # --------------------------------------------------------------------------
    
    # Set up variables for iteration 
    counter = 1;
    a       = [.05, 0.1, 0.36];
    T, K    = size(y);
    
    # Matrix which will save the results
    IRFS = zeros(H+1, K, K, nrep-1);      # nrep-simulations for IRF
    Low  = zeros(H+1, length(a), K, K);   # lower quantiles for each yₜ
    Upp  = zeros(H+1, length(a), K, K);   # upper quantiles for each yₜ
    
    # Bootstrap preliminaries
    data = [-1 1];          # for the random draw 
    U    = u';              # Residuals in TxK dimension
    yᵇ   = zeros(T, K);     # for saving the simulated Yₜ
    A    = B[:,2:end];      # matrix of coefficients for yₜ₋₁, ⋯ , yₜ₋ₚ
    A₀   = B[:,1];          # Intercept
    
    # Bootstrap loop 
    while counter < nrep

        # Random draw to correct for heteroskedasticity w ~ [-1,1]
        w  = rand(data, T);   
        uᵇ = U.*w[p+1:end];  
        
        # the first p-observations are fixed 
        for j=1:K
            for i=1:p
                yᵇ[i, j] = y[i, j];
            end
        end
        
        # Generate bootstrap sample Yₜᵇ for t = p+1,...,T.
        for j = (p+1):T
            xᵇ      = yᵇ[j-1:-1:j-p, :]';                      
            yᵇ[j,:] = A₀ + A*vec(xᵇ) + uᵇ[j-p,:] 
        end
        
        # Compute IRF on bootstrap sample and save results
        IRFS[:,:,:,counter] = IRF_CH(yᵇ, p, H)[1];

        # Update counter 
        counter += 1; 
    end
    
    # Compute quantiles 
    Low = mapslices(u->quantile(u, a./2), IRFS, dims=(4,))
    Upp = mapslices(u->quantile(u, 1.0.-a./2), IRFS, dims=(4,))

    return Low, Upp

end



# ------------------------------------------------------------------------------
# 4 - Estimate IRF by Cholesky 
# ------------------------------------------------------------------------------
# Take the quarterly average
R = get_data(f, "FEDFUNDS", frequency = "q", aggregation_method = "avg").data[:,end-1:end];
u = get_data(f, "UNRATE", frequency = "q", aggregation_method = "avg").data[:,end-1:end];

# Select the insample dates
P   = DataFrame([P[5:end,1] (log.(P[5:end,2]) - log.(P[1:end-4,2]))*100], [:date, :value]);
dic = Dict("R" => R, "u" => u, "P" => P)

for i in 1:length(key)
    aux = findall((dic[key[i]].date .>= Date("01/01/1960","dd/mm/yyyy")) .& 
                  (dic[key[i]].date .<= Date("01/10/2007", "dd/mm/yyyy")));
    dic[key[i]] = dic[key[i]][aux,:];
end

# Specification VAR
y    = [dic["P"].value dic["u"].value dic["R"].value] |> Array{Float64,2};
p    = 4;
H    = 20;
nrep = 1000;

# Estimation SVAR 
IRF, B, U, B₀ = IRF_CH(y, p, H);
Low, Upp      = WILD_CH(y, B, U, p, H, nrep);

# Plot 
x_ax  = collect(0:1:H);
ticks = [0; collect(4:4:H)];
c     = [0.15, 0.25, 0.85];
var   = ["GDP Deflator"; "Unemployment Rate"; "Fed Funds Rate"];
shock = ["Inflation Shock", "Unemployment Shock", "Monetary Policy Shock"];
sav   = ["Chol_inf"; "Chol_un"; "Chol_mon"];


j = 3; k = 3;
plot(size = (700,600), ytickfontsize  = 15, xtickfontsize  = 15,
     xguidefontsize = 15, legendfontsize = 13, boxfontsize = 15,
     framestyle = :box, yguidefontsize = 15, titlefontsize = 20)

# Plot bootstrap confidence interval 
for a in 1:size(Low, 4)
    plot!(x_ax, [IRF[1,j,k]; Low[2:end,j,k,a]], fillrange = [IRF[1,j,k]; Upp[2:end,j,k,a]],
          lw = 1, alpha = c[a], color = "deepskyblue1", xticks = ticks,
          label = "")
end
Plots.plot!(x_ax, IRF[:,j,k], lw = 3, color = "black", xticks = ticks,
            label = var[j])
hline!([0], color = "black", lw = 1, label = nothing)
Plots.plot!(xlabel = "Quarters", ylabel = "", title = shock[k],
                    left_margin = 1Plots.mm, right_margin = 3Plots.mm,
                    bottom_margin = 1Plots.mm, top_margin = 3Plots.mm,
                    xlims = (0,H))
