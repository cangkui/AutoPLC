FUNCTION ERFC : REAL
VAR_INPUT
    X : REAL;
END_VAR
ERFC := 1.0 - ERF(X);

(* revision history
hm	7. apr. 2008	rev 1.0
	original version

hm	10. mar. 2009	rev 1.1
	real constants updated to new systax using dot

*)
END_FUNCTION
