FUNCTION YEAR_END : DATE
VAR_INPUT
    y : INT; // Input variable
END_VAR
YEAR_END := DWORD_TO_DATE(SHR(INT_TO_DWORD(y) * 1461 - 2876712, 2) * 86400);



(* revision history
hm	15. jun. 2008	rev 1.0
	original version

hm	24. jan 2011	rev 1.1
	improved performance
*)
END_FUNCTION
