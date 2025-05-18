FUNCTION ISC_CTRL : BOOL
VAR_INPUT
    IN : BYTE;
END_VAR
ISC_CTRL := IN < 32 OR IN = 127;

(* revision history
hm		6. mar. 2008	rev 1.0
	original version

*)
END_FUNCTION
