FUNCTION INC : INT
VAR_INPUT
    X : INT;
    D : INT;
    M : INT;
END_VAR
INC := (X + D + M + 1) MOD (M + 1);

(* revision history
hm	7. feb 2007		REV 1.0
	original version

hm	15. jan 2008	rev 1.1
	allow for neagtive increment

*)
END_FUNCTION
