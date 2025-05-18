FUNCTION FLOOR : INT
VAR_INPUT
    X : REAL;  
END_VAR
FLOOR := REAL_TO_INT(X);
IF FLOOR > X THEN
	FLOOR := FLOOR - 1;
END_IF;

(* revision history
hm		7. feb 2007		rev 1.0
	originlal version

hm		21. mar. 2008	rev 1.1
	replaced trunc with real_to_int for compatibility reasons

*)
END_FUNCTION
