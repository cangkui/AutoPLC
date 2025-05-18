FUNCTION R2_ADD2 : REAL2
VAR_INPUT
    X : REAL2;
    Y : REAL2;
END_VAR
R2_ADD2.R1 := X.R1 + Y.R1;
R2_ADD2.RX := X.RX + Y.RX;


(* revision history
hm		20.3.2008	rev 1.0
	original version

*)
END_FUNCTION
