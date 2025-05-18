FUNCTION AIR_DENSITY : REAL
VAR_INPUT
    T : REAL;     // Temperature
    P : REAL;     // Pressure
    RH : REAL;    // Relative Humidity
END_VAR

VAR CONSTANT
    RL : REAL := 287.05;   // Constant value for RL
    RX : REAL := 0.003773319;  // Constant value for RX
END_VAR
AIR_DENSITY := OSCAT_BASIC.phys.PN * (1.0 - RH * SDD(T, TRUE) * RX / P) / (RL * ( T - OSCAT_BASIC.phys.T0));

(* original code used for rev 1.0

SDD := SDD(T,FALSE);
RF := RL / (1- RH*SDD/P*RX);
Air_density := P / (RF * (T-TK));

*)

(* revision history

hm	29.10.2007	rev 1.0
	original version

hm	4.12.2007		rev 1.1
	changed code for better performance

hm	18. oct. 2008	rev 1.2
	using phys constants

hm	13. mar. 2009	rev 1.3
	real constants updated to new systax using dot

*)
END_FUNCTION
