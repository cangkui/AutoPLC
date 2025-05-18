FUNCTION_BLOCK HYST_3
VAR_INPUT
    in : REAL;
    hyst : REAL;
    val1 : REAL;
    val2 : REAL;
END_VAR

VAR_OUTPUT
    Q1 : BOOL;
    Q2 : BOOL;
END_VAR

VAR
    X : REAL;
END_VAR
X := hyst * 0.5;
IF in < val1 - X THEN
	q1 := TRUE;
ELSIF in > val1 + X THEN
	q1 := FALSE;
END_IF;
IF in < val2 - X THEN
	q2 := FALSE;
ELSIF in > val2 + X THEN
	q2 := TRUE;
END_IF;



(* revision history
hm	22. jan 2007	rev 1.0
	original version

hm	27. dec 2007	rev 1.1
	changed code for better performance

hm	5. jan 2008	rev 1.2
	further performance iprovements

*)
END_FUNCTION_BLOCK
