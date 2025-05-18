FUNCTION DEW_RH : REAL
VAR_INPUT
    VC : REAL;
    T : REAL;
END_VAR
DEW_RH := LIMIT(0.0, VC / DEW_CON(1.0, T), 100.0);

(* code before rev 1.1
dew_rh := VC / dew_con(100,T) * 100;
IF dew_rh > 100 THEN dew_rh := 100; END_IF;
*)

(* revision history
hm	29. apr 2007	rev 1.0
	original version

hm	2. 12. 2007	rev 1.1
	new code for better performance

hm	13. mar. 2009	rev 1.2
	real constants updated to new systax using dot

*)
END_FUNCTION
