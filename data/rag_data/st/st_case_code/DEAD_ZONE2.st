FUNCTION_BLOCK DEAD_ZONE2
VAR_INPUT
    X : REAL;
    L : REAL;
END_VAR

VAR_OUTPUT
    Y : REAL;
END_VAR
IF ABS(x) > L THEN
	Y := X;
ELSIF Y > 0.0 THEN
	Y := L;
ELSE
	Y := -L;
END_IF;



(* revision history

hm	12. feb. 2007	rev 1.0
	original version

hm	11. mar. 2009	rev 1.1
	real constants updated to new systax using dot

*)
END_FUNCTION_BLOCK
