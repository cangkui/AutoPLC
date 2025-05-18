FUNCTION _ARRAY_INIT : BOOL
VAR_INPUT
    pt : POINTER TO ARRAY [0..32000] OF REAL;
    size : UINT;
    init : REAL;
END_VAR

VAR
    i : UINT;
    stop : UINT;
END_VAR
stop := SHR(size,2)-1;
FOR i := 0 TO stop DO
	pt^[i] := init;
END_FOR;

_array_init := TRUE;

(* revision History

hm 6.1.2007		rev 1.1
	change type of function to bool
	added  array_init := true to set output true.

hm	14.11.2007	rev 1.2
	changed stop calculation to be more efficient

hm	16.3. 2008		rev 1.3
	changed type of input size to uint
*)
END_FUNCTION
