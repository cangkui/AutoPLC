FUNCTION MS_TO_BFT : INT
VAR_INPUT
    MS : REAL;  
END_VAR
MS_TO_BFT := REAL_TO_INT(EXPT(MS * 1.196172, 0.666667));


(* revision history
hm	12. 6. 2008		rev 1.0
	original version

*)
END_FUNCTION
