FUNCTION MIRROR : string
VAR_INPUT
    str: STRING(STRING_LENGTH);
END_VAR

VAR
    pi: POINTER TO ARRAY[1..255] OF BYTE;
    po: POINTER TO BYTE;
    lx: INT;
    i: INT;
END_VAR
pi := ADR(str);
po := ADR(mirror);
lx := LEN(str);
FOR i := lx TO 1 BY - 1 DO
	po^ := pi^[i];
	po := po + 1;
END_FOR;
(* close output string *)
po^:= 0;


(* revision histroy
hm	4. feb. 2008	rev 1.0
	original release

hm	29. mar. 2008	rev 1.1
	changed STRING to STRING(STRING_LENGTH)
*)
END_FUNCTION
