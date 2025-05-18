FUNCTION ISC_NUM : BOOL
VAR_INPUT
    IN : BYTE;
END_VAR
ISC_NUM := IN > 47 AND IN < 58;

(* revision history
hm		6. mar. 2008	rev 1.0
	original version

*)
END_FUNCTION
