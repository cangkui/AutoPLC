FUNCTION BUFFER_SEARCH : INT
VAR_INPUT
    PT : POINTER TO ARRAY[0..32767] OF BYTE;
    SIZE : INT;
    STR : STRING; // STRING_LENGTH should be replaced with actual length if known
    POS : INT;
    IGN : BOOL;
END_VAR

VAR
    ps : POINTER TO ARRAY[0..STRING_LENGTH] OF BYTE; // STRING_LENGTH should be replaced with actual length if known
    chx : BYTE;
    i : INT;
    end : INT;
    k : INT;
    lx : INT;
END_VAR
ps := ADR(STR);
lx := LEN(STR);
end := MIN(SIZE - lx, SIZE - 1);
lx := lx - 1;
FOR i := pos TO end DO
	FOR k := 0 TO lx DO
		IF IGN THEN chx := TO_UPPER(pt^[i+k]); ELSE chx := pt^[i+k]; END_IF;
		IF ps^[k] <> chx THEN EXIT; END_IF;
	END_FOR;
	IF k > lx THEN
		BUFFER_SEARCH := i;
		RETURN;
	END_IF;
END_FOR;
BUFFER_SEARCH := -1;


(* revision History

hm 5. mar. 2008	rev 1.0
	original version

hm	16. mar. 2008	rev 1.1
	chaged type of input size to uint

hm	13. may. 2008	rev 1.2
	changed type of pointer to array[1..32767]
	changed size of string to STRING_LENGTH

hm	12. nov. 2009	rev 1.3
	limit end to array size

hm	25. jan. 2011	rev 1.4
	ign = True will now ignore case
	return -1 if nothing found
*)
END_FUNCTION
