FUNCTION HTML_DECODE : string
VAR_INPUT
    IN : STRING(OSCAT_BASIC.STRING_LENGTH);
END_VAR

VAR
    pos : INT;
    end : INT;
    tmp : STRING(10);
    code : STRING(1);
END_VAR
HTML_DECODE := IN;

(* docode characters encoded by character position *)
pos := FIND(HTML_DECODE, '&');
WHILE pos > 0 DO
	tmp := MID(HTML_DECODE, 2, pos + 1);
	IF tmp = '#x' OR tmp = '#X' THEN
		(* hexadecimal representation found *)
		tmp := MID(HTML_DECODE, 10, pos + 3);
		end := FIND(tmp, ';');
		CODE := OSCAT_BASIC.CHR_TO_STRING(DWORD_TO_BYTE(OSCAT_BASIC.HEX_TO_DWORD(LEFT(tmp, end - 1))));
		HTML_DECODE := REPLACE(HTML_DECODE, CODE, end + 3, pos);
	ELSIF LEFT(tmp,1) = '#' THEN
		tmp := MID(HTML_DECODE, 10, pos + 2);
		end := FIND(tmp, ';');
		CODE := OSCAT_BASIC.CHR_TO_STRING(INT_TO_BYTE(STRING_TO_INT(LEFT(tmp, end - 1))));
		HTML_DECODE := REPLACE(HTML_DECODE, CODE, end + 2, pos);
	ELSE
		(* replace all named characters *)
		tmp := MID(HTML_DECODE, 10, pos + 1);
		end := FIND(tmp, ';');
		CODE := OSCAT_BASIC.CHR_TO_STRING(OSCAT_BASIC.CHARCODE(LEFT(tmp, end - 1)));
		HTML_DECODE := REPLACE(HTML_DECODE, CODE, end + 1, pos);
	END_IF;
	(* search for next occurence of encoded characters *)
	pos := OSCAT_BASIC.FINDP(HTML_DECODE, '&', pos + 1);
END_WHILE;



(* revision history
hm	3. mai. 2008	rev 1.0
	original version

hm	16. jun. 2008	rev 1.1
	changed nested call of replace for better compatibility

hm	17. dec. 2008	rev 1.2
	changed function CHR to CHR_TO_STRING

hm	10. may. 2010	rev 1.3
	stringlength 250

*)
END_FUNCTION
