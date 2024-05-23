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
P = get_data(f, "GDPDEF", frequency = "q").data[:,end-1:end]

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
# 2 - Plot Data
# ------------------------------------------------------------------------------
