FUNCTION IS_NUM : BOOL
VAR_INPUT
    str : STRING[STRING_LENGTH];  // Input variable of type STRING with specified length
END_VAR

VAR
    L : INT;                       // Local variable of type INT
    pt : POINTER TO BYTE;         // Local variable, pointer to BYTE
    pos : INT;                    // Local variable of type INT
END_VAR
PT := ADR(str);
L := LEN(str);
FOR pos := 1 TO L DO
	IF NOT (ISC_NUM(pt^)) THEN
		IS_NUM := FALSE;
		RETURN;
	END_IF;
    	PT := PT + 1;
END_FOR;
IS_NUM := L > 0;


(* revision history
hm	29. feb 2008	rev 1.0
	original version

hm	29. mar. 2008	rev 1.1
	changed STRING to STRING(STRING_LENGTH)
*)
END_FUNCTION
