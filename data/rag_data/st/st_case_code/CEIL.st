FUNCTION CEIL : INT
VAR_INPUT
    X : REAL;
END_VAR
CEIL := REAL_TO_INT(x);
IF CEIL < X THEN
	CEIL := CEIL + 1;
END_IF;


(* revision history
hm	7. feb. 2007	rev 1.0
	original version

hm	21. mar. 2008	rev 1.1
	use REAL_TO_INT instead of trunc for compatibility reasons

*)
END_FUNCTION
