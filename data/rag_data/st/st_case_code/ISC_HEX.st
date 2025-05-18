FUNCTION ISC_HEX : BOOL
VAR_INPUT
    IN : BYTE;
END_VAR
ISC_HEX := (IN > 47 AND IN < 58) OR (IN > 64 AND IN < 71) OR (IN > 96 AND IN < 103);

(* revision history
hm		6. mar. 2008	rev 1.0
	original version

*)
END_FUNCTION
