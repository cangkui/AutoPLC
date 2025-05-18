FUNCTION DEAD_BAND : REAL
VAR_INPUT
    X : REAL;
    L : REAL;
END_VAR
IF X > L THEN
	DEAD_BAND := X - L;
ELSIF X < -L THEN
	DEAD_BAND := X + L;
ELSE
	DEAD_BAND := 0.0;
END_IF;


(* revision history
hm	2. nov. 2007	rev 1.0
	original version

hm	14. jun. 2008	rev 1.1
	improved performance

hm	18. jan. 2011	rev 1.2
	assign 0 before return
*)
END_FUNCTION
