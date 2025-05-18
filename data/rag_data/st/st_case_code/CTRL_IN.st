FUNCTION CTRL_IN : REAL
VAR_INPUT
    SET_POINT : REAL;
    ACTUAL : REAL;
    NOISE : REAL;
END_VAR
(* calculate the process error DIFF *)
CTRL_IN := DEAD_ZONE(SET_POINT - ACTUAL, NOISE);


(* revision history
hm 	2. jun. 2008 	rev 1.0
	original version

*)
END_FUNCTION
