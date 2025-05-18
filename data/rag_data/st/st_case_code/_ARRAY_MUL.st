FUNCTION _ARRAY_MUL : BOOL
VAR_INPUT
    pt : POINTER TO ARRAY [0..32000] OF REAL;
    size : UINT;
    X : REAL;
END_VAR

VAR
    i : UINT;
    stop : UINT;
END_VAR
stop := SHR(size,2)-1;
FOR i := 0 TO stop DO
	PT^[i] := PT^[i] * X;
END_FOR;
_ARRAY_MUL := TRUE;


(* revision history
hm	2. apr 2008		rev 1.0
	original version

*)
END_FUNCTION
