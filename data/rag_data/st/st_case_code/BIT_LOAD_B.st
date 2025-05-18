FUNCTION BIT_LOAD_B : BYTE
VAR_INPUT
    IN : BYTE;
    VAL : BOOL;
    POS : INT;
END_VAR

VAR CONSTANT
    dat : BYTE := 1;
END_VAR
IF VAL THEN
	BIT_LOAD_B := in OR SHL(dat,pos);
ELSE
	BIT_LOAD_B := in AND (NOT SHL(dat,pos));
END_IF;


(* revision history
hm	29. feb 2008	rev 1.0
	original version

hm	16. mar 2008	rev 1.1
	change input bit to val for compatibility reasons
*)
END_FUNCTION
