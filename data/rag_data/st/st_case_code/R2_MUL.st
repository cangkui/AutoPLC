FUNCTION R2_MUL : REAL2
VAR_INPUT
    X : REAL2;
    Y : REAL;
END_VAR
R2_MUL.RX := X.RX * Y;
R2_MUL.R1 := X.R1 * Y;


(* revision history
hm		20.3.2008	rev 1.0
	original version

*)
END_FUNCTION
