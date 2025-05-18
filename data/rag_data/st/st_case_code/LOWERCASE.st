FUNCTION LOWERCASE : string
VAR_INPUT
    str: STRING[STRING_LENGTH];
END_VAR

VAR
    PT: POINTER TO BYTE;
    pos: INT;
    l: INT;
END_VAR
PT := ADR(LOWERCASE);
lowercase := str;
l := LEN(str);
FOR pos := 1 TO l DO
	pt^ := TO_LOWER(pt^);
	PT := PT + 1;
END_FOR;

(* revision histroy
hm	6. oct. 2006	rev 1.0
	original release

hm	4. feb. 2008	rev 1.1
	improved performance
	added Ä.Ö,Ü

hm	6. mar. 2008	rev 1.2
	added support for exteded Ascii

hm	29. mar. 2008	rev 1.3
	changed STRING to STRING(STRING_LENGTH)
*)
END_FUNCTION
