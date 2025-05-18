FUNCTION D_TRUNC : DINT
VAR_INPUT
    X : REAL;
END_VAR
D_TRUNC := REAL_TO_DINT(X);
IF X > 0.0 THEN
	IF DINT_TO_REAL(D_TRUNC) > X THEN D_TRUNC := D_TRUNC - 1; END_IF;
ELSE
	IF DINT_TO_REAL(D_TRUNC) < X THEN D_TRUNC := D_TRUNC + 1; END_IF;
END_IF;


(* for systems that support a dint truncation this routine can be replaced by trunc() *)


(* revision history
hm	21. mar. 2008	rev 1.0
	original version

hm	31. oct. 2008	rev 1.1
	optimized performance

hm	10. mar. 2009	rev 1.2
	real constants updated to new systax using dot
*)
END_FUNCTION
