FUNCTION CACOS : complex
VAR_INPUT
    X : COMPLEX;
END_VAR

VAR
    Y : COMPLEX;
END_VAR
y := CACOSH(x);
CACOS.re := y.im;
CACOS.im := -y.re;



(* revision history
hm	21. feb 2008	rev 1.0
	original version


*)
END_FUNCTION
