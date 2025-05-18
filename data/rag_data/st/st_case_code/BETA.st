FUNCTION BETA : REAL
VAR_INPUT
    X : REAL;
    Y : REAL;
END_VAR
BETA := GAMMA(X) * GAMMA(Y) / GAMMA(x + y);


(* revision history
hm	26. oct. 2008	rev 1.0
	original version

*)
END_FUNCTION
