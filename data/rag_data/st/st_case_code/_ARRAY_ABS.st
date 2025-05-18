FUNCTION _ARRAY_ABS : BOOL
VAR_INPUT
    pt : POINTER TO ARRAY[0..32000] OF REAL;
    size : UINT;
END_VAR

VAR_LOCAL
    i : UINT;
    stop : UINT;
END_VAR
stop := SHR(size,2)-1;
FOR i := 0 TO stop DO
	PT^[i] := ABS(PT^[i]);
END_FOR;
_ARRAY_ABS := TRUE;


(* revision history
hm	2. apr 2008		rev 1.0
	original version

*)
END_FUNCTION
