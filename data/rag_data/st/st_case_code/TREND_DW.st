FUNCTION_BLOCK TREND_DW
VAR_INPUT
    X : DWORD;
END_VAR
VAR_OUTPUT
    Q : BOOL;
    TU : BOOL;
    TD : BOOL;
    D : DWORD;
END_VAR
VAR
    last_X : DWORD;
END_VAR
IF X > last_X THEN
	TU := TRUE;
	TD := FALSE;
	D := X - last_X;
	Q := TRUE;
ELSIF X < last_X THEN
	TD := TRUE;
	TU := FALSE;
	D := last_X - X;
	Q := FALSE;
ELSE
	TU := FALSE;
	TD := FALSE;
	D := 0;
END_IF;
last_X := X;



(* revision history
hm	21. nov. 2008	rev 1.0
	original version

hm	20. feb. 2009	rev 1.1
	added outputs TU, TD and D

hm	14. mar. 2009	rev 1.2
	removed double assignments

*)
END_FUNCTION_BLOCK
