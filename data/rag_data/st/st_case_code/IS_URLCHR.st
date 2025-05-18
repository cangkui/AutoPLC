FUNCTION IS_URLCHR : BOOL
VAR_INPUT
    IN : BYTE;
END_VAR
IS_URLCHR := ((IN > 47) AND (IN < 58)) OR ((IN > 64) AND (IN < 91)) OR ((IN > 96) AND (IN < 123)) OR (in = 126) OR (in = 95) OR (in =45) OR (in =46);


(* revision history
hm		27. apr. 2008	rev 1.0
	original version

*)
END_FUNCTION
