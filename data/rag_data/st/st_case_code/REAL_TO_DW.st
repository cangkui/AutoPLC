FUNCTION REAL_TO_DW : DWORD
VAR_INPUT
    X : REAL;  // Input variable of type REAL
END_VAR

VAR
    pt : POINTER TO DWORD;  // Local variable of pointer type to DWORD
END_VAR
pt := ADR(X);
REAL_TO_DW := pt^;

(* revision history
hm	18. apr. 2008
	original version

*)
END_FUNCTION
