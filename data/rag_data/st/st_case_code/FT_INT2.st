FUNCTION_BLOCK FT_INT2
VAR_INPUT
    IN : REAL;
    K : REAL := 1.0;
    RUN : BOOL := TRUE;
    RST : BOOL;
    OUT_MIN : REAL := -1E+38;
    OUT_MAX : REAL := 1E+38;
END_VAR

VAR_OUTPUT
    OUT : REAL;
    LIM : BOOL;
END_VAR

VAR
    integ : INTEGRATE;
    ix : REAL;
    val : REAL2;
END_VAR
IF RST THEN
	val := R2_SET(0.0);
	out := 0.0;
ELSE
	integ(X := IN, E := RUN, K := K, Y := ix);
	val := R2_ADD(val, ix);
	ix := 0.0;
	OUT := val.RX;
END_IF;

(* check output for limits *)
IF out > OUT_MIN AND out < OUT_MAX THEN
	LIM := FALSE;
ELSE
	OUT := LIMIT(OUT_MIN, OUT, OUT_MAX);
	val := R2_SET(OUT);
	LIM := TRUE;
END_IF;



(*	revision history
hm	2. jun. 2008	rev 1.0
	original version

hm	5. nov. 2008	rev 1.1
	rewritten code using integrate

hm	11. mar. 2009	rev 1.2
	real constants updated to new systax using dot

*)
END_FUNCTION_BLOCK
