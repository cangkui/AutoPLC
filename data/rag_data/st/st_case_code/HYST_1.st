FUNCTION_BLOCK HYST_1
VAR_INPUT
    In : REAL;
    high : REAL;
    low : REAL;
END_VAR_INPUT

VAR_OUTPUT
    Q : BOOL;
    win : BOOL;
END_VAR_OUTPUT
IF in < low THEN
	Q := FALSE;
	win := FALSE;
ELSIF in > high THEN
	Q := TRUE;
	win := FALSE;
ELSE
	win := TRUE;
END_IF;




(* revision history
hm		4.  aug. 2006	rev 1.0
	original version

hm		2. jun. 2008	rev 1.1
	improved performance
*)
END_FUNCTION_BLOCK
