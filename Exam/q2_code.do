clear all

cd "C:\Users\lapobini\Desktop"

import delimited "data", clear

gen stata_date = date(date, "YMD")  // Convert to Stata date
gen quarterly_date = qofd(stata_date)  // Convert to quarterly date
tsset quarterly_date

* Format the quarterly date to display in "YYYYqQ" format
format quarterly_date %tq

* Impulse 1
var pi gap, lags(1/8) exog(L(0/12).i)
irf set results1, replace
irf create irf1, step(16) set(results1) replace
irf graph dm, impulse(i) irf(irf1) 
graph export "irf1.pdf", replace

* Impulse 2
var pi gap, lags(1/8) exog(L(0/12).rr)
irf create results2, step(16) replace
irf graph dm, impulse(rr) irf(results2) 
graph export "irf2.pdf", replace