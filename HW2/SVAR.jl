# ------------------------------------------------------------------------------
# SVAR
# ------------------------------------------------------------------------------
# Author: Lapo Bini, lbini@ucsd.edu
# Problem Set 2 - Prof. Johannes Wieland - 22/15/2024
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
using DataFrames, LinearAlgebra, Dates, Statistics, Plots, LaTeXStrings
using CSV, XLSX, FredData, HTTP


# ------------------------------------------------------------------------------
# Download FRED Data
# ------------------------------------------------------------------------------
# Establish connection with FRED API (Do not share API key)

api_key   = "66c080f0ed7880e7df1230ef212fb8c1";
f         = Fred(api_key);


# ------------------------------------------------------------------------------
#                                DOWNLOAD DATA
# ------------------------------------------------------------------------------
@info("FRED_data > downloading series")
# US Regular All Formulations Gas Price, Dollars per Gallon (LEVEL)
gasregw = get_data(f, "GASREGW", frequency = "w")
GASREGW = gasregw.data[:,end-1:end];

# Crude Oil Prices: West Texas Intermediate (WTI) - (YoY - daily to Weekly EOP)
dcoilwtico_ch1 = get_data(f, "DCOILWTICO", frequency = "wesa",
                          aggregation_method = "eop", units = "ch1")
DCOILWTICO_CH1 = dcoilwtico_ch1.data[:,end-1:end];

# US Regular Conventional Gas Price, Dollars per Gallon (LEVEL)

gasregcovw = get_data(f, "GASREGCOVW", frequency = "w")
GASREGCOVW = gasregcovw.data[:,end-1:end];

# Crude Oil Prices: West Texas Intermediate (WTI) - (YoY - daily to Weekly EOP)
dcoilbrenteu_ch1 = get_data(f, "DCOILBRENTEU", frequency = "wesa",
                          aggregation_method = "eop", units = "ch1")
DCOILBRENTEU_CH1 = dcoilbrenteu_ch1.data[:,end-1:end];

# US Diesel Sales Price - (already YoY)
gasdesw_pc1 = get_data(f, "GASDESW", frequency = "w", units = "pc1");
GASDESW_PC1 = gasdesw_pc1.data[:,end-1:end];

# US Midgrade All Formulations Gas Price - already (YoY)

gasmidw_pc1 = get_data(f, "GASMIDW", frequency = "w", units = "pc1");
GASMIDW_PC1 = gasmidw_pc1.data[:,end-1:end];

# PADD V (West Coast District) Diesel Sales Price - already (YoY)


gasdeswcw_pc1 = get_data(f, "GASDESWCW", frequency = "w", units = "pc1");
GASDESWCW_PC1 = gasdeswcw_pc1.data[:,end-1:end];

# PADD I (East Coast District) Diesel Sales Price - already (YoY)

gasdesecw_pc1 = get_data(f, "GASDESECW", frequency = "w", units = "pc1");
GASDESECW_PC1 = gasdesecw_pc1.data[:,end-1:end];

# PADD V (West Coast District) Regular Conventional Gas Price - already (YoY)

gasregcovwcw_ch1 = get_data(f, "GASREGCOVWCW", frequency = "w", units = "ch1");
GASREGCOVWCW_CH1 = gasregcovwcw_ch1.data[:,end-1:end];


# ------------------------------------------------------------------------------
#                               SAVE FILE
# ------------------------------------------------------------------------------

@info("FRED_data > saving FRED_most_updated.xlsx file")
series_tickers = ["DCOILWTICO_CH1", "GASREGW", "GASREGCOVW", "GASDESW_PC1", "GASMIDW_PC1",
                  "GASDESWCW_PC1", "GASDESECW_PC1", "DCOILBRENTEU_CH1", "GASREGCOVWCW_CH1"];
# Dataset name
data_frame = Dict();
data_frame["DCOILWTICO_CH1"]   = DCOILWTICO_CH1;
data_frame["GASREGW"]          = GASREGW;
data_frame["GASREGCOVW"]       = GASREGCOVW;
data_frame["GASDESW_PC1"]      = GASDESW_PC1;
data_frame["GASMIDW_PC1"]      = GASMIDW_PC1;
data_frame["GASDESWCW_PC1"]    = GASDESWCW_PC1;
data_frame["GASDESECW_PC1"]    = GASDESECW_PC1;
data_frame["DCOILBRENTEU_CH1"] = DCOILBRENTEU_CH1;
data_frame["GASREGCOVWCW_CH1"] = GASREGCOVWCW_CH1;

# Is there already a dataset?
df_name = "FRED_most_updated.xlsx";
df      = DataFrame(Intro = "Last update: "*string(Dates.now()))
XLSX.writetable(df_name, df, overwrite = true);

XLSX.openxlsx(df_name, mode = "rw") do xf
    for i in 1:length(series_tickers)
        XLSX.addsheet!(xf, series_tickers[i])
        sheet       = xf[i+1]
        sheet["A1"] = [data_frame[series_tickers[i]].date data_frame[series_tickers[i]].value];
    end
end
@info("FRED_data > done > FRED_most_updated.xlsx has been updated")
@warn(" 1 - Be careful with the length of the sample and the last day of the week")
@warn(" 2 - Missing values saved as NaN, you should delete all of them")
