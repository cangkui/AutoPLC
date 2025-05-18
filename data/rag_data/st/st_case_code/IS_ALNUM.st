FUNCTION IS_ALNUM : BOOL
VAR_INPUT
    str : STRING(STRING_LENGTH);
END_VAR_INPUT

VAR
    L : INT;
    pt : POINTER TO BYTE;
    pos : INT;
END_VAR
PT := ADR(str);
L := LEN(str);
FOR pos := 1 TO L DO
	IF NOT (ISC_ALPHA(pt^) OR ISC_NUM(pt^)) THEN
		IS_ALNUM := FALSE;
		RETURN;
	END_IF;
    	PT := PT + 1;
END_FOR;
IS_ALNUM := L > 0;


(* revision history
hm	29. feb 2008	rev 1.0
	original version

hm	29. mar. 2008	rev 1.1
	changed STRING to STRING(STRING_LENGTH)
*)
END_FUNCTION
