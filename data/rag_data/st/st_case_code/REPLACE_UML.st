FUNCTION REPLACE_UML : string
VAR_INPUT
    str : STRING; // STRING_LENGTH is not defined in XML, use default STRING
END_VAR

VAR
    L : INT;
    pt : POINTER TO BYTE;
    pto : POINTER TO BYTE;
    ptm : POINTER TO BYTE;
    pt1 : POINTER TO BYTE;
    pt2 : POINTER TO BYTE;
    su : STRING[2];
    pos : INT;
END_VAR
PT := ADR(str);
pto := ADR(REPLACE_UML);
ptm := pto + INT_TO_DWORD(string_length);
pt1 := ADR(su);
pt2 := pt1 + 1;
L := LEN(str);
WHILE pos < L AND pos < string_length DO
	IF pt^ < 127 THEN
		(* no uml character simlply copy the character*)
		pto^ := pt^;
	ELSE
		(* convert the uml character *)
		su := TO_UML(pt^);
		(* we must make sure pointer are not out of range *)
		pto^ := pt1^;
		IF pto < ptm AND pt2^ > 0 THEN
			pto := pto + 1;
			pto^ := pt2^;
		END_IF;
	END_IF;
	(* increment pointers *)
	pt := pt + 1;
	pto := pto + 1;
	pos := pos + 1;
END_WHILE;

(* properly close the output string *)
pto^ := 0;


(* revision history
hm	29. feb 2008	rev 1.0
	original version

hm	29. mar. 2008	rev 1.1
	changed STRING to STRING(STRING_LENGTH)
	new code to avoid pointer out of range
	use new function to_uml
*)
END_FUNCTION
