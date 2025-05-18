FUNCTION CPOL : complex
VAR_INPUT
    L: REAL;
    A: REAL;
END_VAR
CPOL.re := L * COS(A);
CPOL.im := L * SIN(A);


(* revision history
hm	21. feb 2008	rev 1.0
	original version

*)
END_FUNCTION
