FUNCTION IS_LOWER : BOOL
VAR_INPUT
    str : STRING[STRING_LENGTH]; // Input variable declaration
END_VAR

VAR
    l : INT; // Local variable declaration
    pt : POINTER TO BYTE; // Pointer variable declaration
    pos : INT; // Position variable declaration
END_VAR
PT := ADR(str);
L := LEN(str);
FOR pos := 1 TO L DO
	IF NOT (ISC_LOWER(pt^)) THEN
		IS_LOWER := FALSE;
		RETURN;
	END_IF;
    	PT := PT + 1;
END_FOR;
IS_LOWER := L > 0;


(* revision history
hm	29. feb 2008	rev 1.0
	original version

hm	29. mar. 2008	rev 1.1
	changed STRING to STRING(STRING_LENGTH)
*)
END_FUNCTION
