FUNCTION BIT_TOGGLE_W : WORD
VAR_INPUT
    IN : WORD;
    POS : INT;
END_VAR
BIT_TOGGLE_W := SHL(WORD#1, POS) XOR IN;



(* revision history
hm	18. oct. 2008	rev 1.0
	original version


*)
END_FUNCTION
