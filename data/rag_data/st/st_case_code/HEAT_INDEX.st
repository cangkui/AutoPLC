FUNCTION HEAT_INDEX : REAL
VAR_INPUT
    T : REAL;
    RH : REAL;
END_VAR

VAR
    RH2 : REAL;
    T2 : REAL;
END_VAR
IF RH < 20.0 OR T < 20.0 THEN
	HEAT_INDEX := T;
ELSE
	RH2 := RH * RH;
	T := OSCAT_BASIC.C_TO_F(T);
	T2 := T * T;
	HEAT_INDEX := -42.379 + 2.04901523 * T - 6.83783E-3 * T2
			+ RH * (10.1433127 - 0.22475541 * T + 1.22874E-3 * T2)
			+ RH2 * (8.5282E-4 * T - 5.481717E-2 - 1.99E-6 * T2);
	HEAT_INDEX := OSCAT_BASIC.F_TO_C(HEAT_INDEX);
END_IF;

(* revision history

hm 27.2.2007		rev 1.1
	deleted unused variables rh3 and T3

hm	2.12.2007		rev 1.2
	changed code for better performance

hm	13. mar. 2009	rev 1.3
	real constants updated to new systax using dot

*)
END_FUNCTION
