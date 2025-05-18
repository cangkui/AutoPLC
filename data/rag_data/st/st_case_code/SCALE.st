FUNCTION SCALE : REAL
VAR_INPUT
    X : REAL;
    K : REAL;
    O : REAL;
    MX : REAL;
    MN : REAL;
END_VAR
SCALE := LIMIT(MN, X * K + O, MX);


(* revision history
hm	16. may. 2008		rev 1.0
	original version
*)
END_FUNCTION
