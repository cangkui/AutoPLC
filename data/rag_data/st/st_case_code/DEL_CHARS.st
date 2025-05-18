FUNCTION DEL_CHARS : string
VAR_INPUT
    IN : STRING; // Length not specified in ST code
    CX : STRING[80];
END_VAR

VAR
    pos : INT := 1;
    stop : INT;
END_VAR
(* copy input string *)
DEL_CHARS := IN;
stop := LEN(in);
WHILE pos <= stop DO
	IF FIND(cx, MID(DEL_CHARS, 1, pos)) > 0 THEN
		(* wrong chracter needs to be deleted *)
		DEL_CHARS := DELETE(DEL_CHARS, 1, pos);
		stop := stop - 1;	(* the string is one character shorter now *)
	ELSE
		(* charcter not found skip to next one *)
		pos := pos + 1;
	END_IF;
END_WHILE;



(* revision history

hm		18. jun. 2008	rev 1.0		
	original version 


*)
END_FUNCTION
