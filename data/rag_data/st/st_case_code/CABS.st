FUNCTION CABS : REAL
VAR_INPUT
    X : COMPLEX;
END_VAR
CABS := HYPOT(x.re, x.im);


(* revision history
hm	21. feb 2008	rev 1.0
	original version

*)
END_FUNCTION
