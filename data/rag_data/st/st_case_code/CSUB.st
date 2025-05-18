FUNCTION CSUB : complex
VAR_INPUT
    X : COMPLEX;
    Y : COMPLEX;
END_VAR
CSUB.re := X.re - Y.re;
CSUB.im := X.im - Y.im;

(* revision history
hm	21. feb 2008	rev 1.0
	original version

*)
END_FUNCTION
