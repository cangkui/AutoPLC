FUNCTION INC2 : INT
VAR_INPUT
    X : INT;
    D : INT;
    L : INT;
    U : INT;
END_VAR

VAR
    tmp : INT;
END_VAR
tmp := U - L + 1;
INC2 := (X + D - L + tmp) MOD tmp + L;



(* revision history
hm	29. jun. 2008		REV 1.0
	original version


*)
END_FUNCTION
