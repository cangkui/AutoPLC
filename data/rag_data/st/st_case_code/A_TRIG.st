FUNCTION_BLOCK A_TRIG
VAR_INPUT
    IN : REAL;
    RES : REAL;
END_VAR

VAR_OUTPUT
    Q : BOOL;
    D : REAL;
END_VAR

VAR
    last_in : REAL;
END_VAR
D := IN - LAST_IN;
Q := ABS(D) > res;
IF Q THEN last_in := IN; END_IF;
D := IN - LAST_IN;


(* revision history

hm 	16. jul. 2008	rev 1.0
	original version released

hm	25. oct. 2008	rev 1.1
	code optimization
	a_trig now also works for res = 0

*)
END_FUNCTION_BLOCK
