FUNCTION IS_NCC : BOOL
VAR_INPUT
    str : STRING[STRING_LENGTH];
    cmp : STRING[STRING_LENGTH];
END_VAR

VAR
    L : INT;
    pos : INT;
END_VAR
L := LEN(str);
FOR pos := 1 TO L DO
	IF FIND(CMP,MID(str,1,pos)) > 0 THEN RETURN; END_IF;
END_FOR;
IS_NCC := TRUE;



(* revision history
hm	19. mar 2008	rev 1.0
	original version

hm	29. mar. 2008	rev 1.1
	changed STRING to STRING(STRING_LENGTH)
*)
END_FUNCTION
