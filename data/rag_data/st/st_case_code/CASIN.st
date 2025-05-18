FUNCTION CASIN : complex
VAR_INPUT
    X : complex; (* Input variable of type complex *)
END_VAR

VAR
    Y : complex; (* Local variable of type complex *)
END_VAR
y.re := -x.im;
y.im := x.re;
y := CASINH(y);
CASIN.re := y.im;
CASIN.im := -y.re;



(* revision history
hm	21. feb 2008	rev 1.0
	original version


*)
END_FUNCTION
