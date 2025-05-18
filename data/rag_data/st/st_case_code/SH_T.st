FUNCTION_BLOCK SH_T
VAR_INPUT
    IN : REAL;
    E : BOOL;
END_VAR

VAR_OUTPUT
    OUT : REAL;
END_VAR
IF E THEN out := in; END_IF;



(* revision history
hm	1. sep. 2006	rev 1.0
	original version

hm	18. oct. 2008	rev 1.1
	changed input en to e for compatibility reasons

*)
END_FUNCTION_BLOCK
