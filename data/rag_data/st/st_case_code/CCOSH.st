FUNCTION CCOSH : complex
VAR_INPUT
    X : COMPLEX;
END_VAR
CCOSH.re := cosH(x.re) * COS(x.im);
CCOSH.im := sinH(x.re) * SIN(x.im);


(* revision history
hm	21. feb 2008	rev 1.0
	original version

*)
END_FUNCTION
