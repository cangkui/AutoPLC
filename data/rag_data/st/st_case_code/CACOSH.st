FUNCTION CACOSH : complex
VAR_INPUT
    X : COMPLEX; // Input variable
END_VAR

VAR
    Y : COMPLEX; // Local variable
END_VAR
y.re := (X.re - X.im)*(X.re + X.im) - 1.0;
y.im := 2.0 * X.re * X.im;
y    := CSQRT(y);
y.re := y.re + x.re;
y.im := y.im + x.im;
CACOSH  := CLOG(y);



(* revision history
hm	21. feb 2008	rev 1.0
	original version


*)
END_FUNCTION
