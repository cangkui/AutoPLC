FUNCTION CINV : complex
VAR_INPUT
    X : COMPLEX; // Input variable of type complex
END_VAR

VAR
    temp : REAL; // Local variable of type REAL
END_VAR
temp := X.re * X.re + X.im * X.im;
CINV.re := X.re / temp;
CINV.im := -X.im / temp;

(* revision history
hm	21. feb 2008	rev 1.0
	original version

*)
END_FUNCTION
