FUNCTION CLOG : complex
VAR_INPUT
    X : COMPLEX; // Derived type 'complex'
END_VAR
CLOG.re := LN(HYPOT(X.re, X.im));
CLOG.im := ATAN2(X.im, X.re);


(* revision history
hm	21. feb 2008	rev 1.0
	original version

hm	20. apr. 2008	rev 1.1
	use ATAN2 instead of ATAN
*)
END_FUNCTION
