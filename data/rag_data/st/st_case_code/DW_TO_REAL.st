FUNCTION DW_TO_REAL : REAL
VAR_INPUT
    X : DWORD; // Input variable
END_VAR

VAR
    pt : ^REAL; // Pointer to REAL type
END_VAR
pt := ADR(X);
DW_TO_REAL := pt^;

(* revision history
hm	18. apr. 2008
	original version

*)
END_FUNCTION
