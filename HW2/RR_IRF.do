
********************************************************************************
clear all
set scheme s1color 
graph close _all
********************************************************************************
*FILE PATHS
********************************************************************************
local datain "./210C/HW2/" 
local dataout "./210C/HW2/" 
local results "./210C/HW2/" 
********************************************************************************
{ 

use `datain'RR_monetary_shock_quarterly.dta, clear 

rename date qdate 
format qdate %tq 

tempfile romer
save `romer', replace 
	
	
*lOAD DATA FROM FRED
local var "FEDFUNDS UNRATE GDPDEF USRECM"
	
import fred `var' , clear 

*GENERATE YOY INFLATION
gen INFL = ((GDPDEF - l12.GDPDEF)/l12.GDPDEF)*100
		

* LABEL VARS
lab var FEDFUNDS "Federal Funds Effective Rate"
lab var UNRATE   "Unemployment Rate"
lab var GDPDEF   "GDP Deflator"
lab var INFL	 "Inflation Rate"

		
*Restrict analysis to before the Great Recession (1C)
keep if qdate >= tq(1960q1) & qdate <= tq(2007q4)

********************************************************************************
*Question 2B
********************************************************************************

tsset qdate 
var INFL UNRATE FEDFUNDS, lags(1/8) exog(L(0/12).resid_full)
irf create myrirf, step(20) replace
irf graph dm, impulse(resid_full) irf(myrirf) byopts(title(VAR with 8 Lags and RR Shocks) yrescale) /// INFL UNRATE 
yline(0,  lcolor(black) lp(dash) lw(*2)) legend(col(2) order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
name(var_results, replace )
graph export `results'var_irf_RRshock.pdf, replace

stop 
