FUNCTION FLOOR2 : DINT
VAR_INPUT
    X : REAL;
END_VAR
FLOOR2 := REAL_TO_DINT(X);
IF DINT_TO_REAL(FLOOR2) > X THEN
	FLOOR2 := FLOOR2 - 1;
END_IF;

(* revision history
hm	21. mar. 2008	rev 1.0
	originlal version

hm	4. apr. 2008	rev 1.1
	added type conversion to avoid warnings under codesys 3.0
*)
END_FUNCTION
