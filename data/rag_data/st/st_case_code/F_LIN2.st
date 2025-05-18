FUNCTION F_LIN2 : REAL
VAR_INPUT
    X   : REAL;
    X1  : REAL;
    Y1  : REAL;
    X2  : REAL;
    Y2  : REAL;
END_VAR
F_LIN2 := (Y2 - Y1) / (X2 - X1) * (X - X1) + Y1;


(* revision history
hm	1. jan. 2007	rev 1.0
	original release

hm	17. dec. 2008	rev 1.1
	optimized formula

*)
END_FUNCTION
