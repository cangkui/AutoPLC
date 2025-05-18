FUNCTION URL_DECODE : string
VAR_INPUT
    IN : STRING(OSCAT_BASIC.STRING_LENGTH);
END_VAR

VAR
    pos : INT;
    seq : STRING(3);
    new : STRING(1);
END_VAR
URL_DECODE := IN;
pos := FIND(URL_DECODE,'%');
WHILE pos > 0 DO
	seq := MID(URL_DECODE,2,pos + 1);
	new := OSCAT_BASIC.CHR_TO_STRING(DWORD_TO_BYTE(OSCAT_BASIC.HEX_TO_DWORD(seq)));
	URL_DECODE := REPLACE(URL_DECODE, new, 3, pos);
	pos := FIND(URL_DECODE,'%');
END_WHILE;


(* revision history
hm	27. apr. 2008	rev 1.0
	original version

hm	17. dec. 2008	rev 1.1
	changes name of function chr to chr_to_string

ks	10. mai. 2008	rev 1.2
	stringlength 250
*)
END_FUNCTION
