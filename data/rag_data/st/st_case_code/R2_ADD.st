FUNCTION R2_ADD : REAL2
VAR_INPUT
    X : Real2;  // Derived type
    Y : REAL;   
END_VAR

VAR
    temp : REAL; 
END_VAR
temp := X.RX;
R2_ADD.RX := Y + X.R1 + X.RX;
R2_ADD.R1 := temp - R2_ADD.RX + Y + X.R1;


(* revision history
hm		20.3.2008	rev 1.0
	original version

*)
END_FUNCTION
