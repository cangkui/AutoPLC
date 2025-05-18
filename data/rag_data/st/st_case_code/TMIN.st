FUNCTION_BLOCK TMIN
VAR_INPUT
    IN : BOOL;
    PT : TIME;
END_VAR

VAR_OUTPUT
    Q : BOOL;
END_VAR

VAR
    pm : TP;  // Assuming TP is a previously defined derived type
END_VAR
pm(in := IN, PT := PT);
Q := IN OR pm.Q;


(* revision history
hm	21. jul. 2008	rev 1.0
	original version

*)
END_FUNCTION_BLOCK
