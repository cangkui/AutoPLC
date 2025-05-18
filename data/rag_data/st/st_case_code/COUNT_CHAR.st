FUNCTION COUNT_CHAR : INT
VAR_INPUT
    str : STRING[STRING_LENGTH];  // Input variable of type STRING with the specified length
    chr : BYTE;                    // Input variable of type BYTE
END_VAR

VAR
    l : INT;                       // Local variable of type INT
    pt : POINTER TO BYTE;          // Local variable that is a pointer to BYTE
    pos : INT;                     // Local variable of type INT
END_VAR
PT := ADR(str);
l := LEN(str);
COUNT_CHAR := 0;
FOR pos := 1 TO l DO
	IF pt^ = CHR THEN COUNT_CHAR := COUNT_CHAR + 1; END_IF;
    PT := PT + 1;
END_FOR;


(* revision history
hm	29. feb 2008	rev 1.0
	original version

hm	29. mar. 2008	rev 1.1
	changed STRING to STRING(STRING_LENGTH)
*)
END_FUNCTION
