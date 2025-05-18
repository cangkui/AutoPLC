FUNCTION TRIM : string
VAR_INPUT
    str : STRING[STRING_LENGTH];
END_VAR_INPUT

VAR
    pos : INT;
END_VAR
TRIM := str;
REPEAT
	pos := FIND(trim,' ');
	IF pos > 0 THEN TRIM := REPLACE(TRIM,'',1,pos); END_IF;
UNTIL pos = 0	END_REPEAT;


(* revision histroy
hm	6.10.2006		rev 1.0
	original release

hm	20. mar. 2008	rev 1.1
	avoid to call replace with pos = 0 because some systems will produce an error

hm	29. mar. 2008	rev 1.2
	changed STRING to STRING(STRING_LENGTH)
*)
END_FUNCTION
