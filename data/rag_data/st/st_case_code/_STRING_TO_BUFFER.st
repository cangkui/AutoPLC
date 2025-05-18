FUNCTION _STRING_TO_BUFFER : INT
VAR_INPUT
    STR : STRING[STRING_LENGTH];
    POS : INT;
    PT : POINTER TO ARRAY[0..32767] OF BYTE;
    SIZE : UINT;
END_VAR_INPUT

VAR
    ps : POINTER TO BYTE;
    i : INT;
    end : INT;
END_VAR
ps := ADR(str);
end := MIN(pos + LEN(str), UINT_TO_INT(size));
IF end > 0 THEN end := end -1; END_IF;
FOR i := pos TO end DO
	pt^[i] := ps^;
	ps := ps + 1;
END_FOR;

_STRING_TO_BUFFER := i;

(* revision History

hm 	5. mar. 2008	rev 1.0
	original version

hm	16. mar. 2008	rev 1.1
	changed type of input size to uint

hm	13. may. 2008	rev 1.2
	changed type of pointer to array[1..32767]
	changed size of string to STRING_LENGTH

hm	12. nov. 2009	rev 1.3
	limit end to size - 1

hm	2. jan 2012	rev 1.4
	return the position after the input string when finished
*)
END_FUNCTION
