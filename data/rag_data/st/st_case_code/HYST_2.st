FUNCTION_BLOCK HYST_2
VAR_INPUT
    IN   : REAL;
    VAL  : REAL;
    HYS  : REAL;
END_VAR

VAR_OUTPUT
    Q    : BOOL;
    WIN  : BOOL;
END_VAR

VAR
    tmp  : REAL;
END_VAR
tmp := val - hys * 0.5;
IF in < tmp THEN
	Q := FALSE;
	win := FALSE;
ELSIF in > tmp + hys THEN
	Q := TRUE;
	win := FALSE;
ELSE
	win := TRUE;
END_IF;



(* revision history
hm		4. aug 2006	rev 1.0
	original version

hm		5. jan 2008	rev 1.1
	improved code for better performance

hm		2. jun. 2008	rev 1.2
	improved performance
*)
END_FUNCTION_BLOCK
