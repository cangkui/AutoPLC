FUNCTION IP4_DECODE : DWORD
VAR_INPUT
    str : STRING(15);
END_VAR

VAR
    pos : INT;
END_VAR
pos := FIND(str, '.');
WHILE pos > 0 DO
	IP4_DECODE := SHL(IP4_DECODE,8) OR STRING_TO_DWORD(LEFT(str,pos - 1));
	str := DELETE(str, pos, 1);
	pos := FIND(str, '.');
END_WHILE;
IP4_DECODE := SHL(IP4_DECODE,8) OR STRING_TO_DWORD(str);


(* revision history
hm		29. feb 2008	rev 1.0
	original version

hm	10. mar. 2009	rev 1.1
	new algorythm
*)
END_FUNCTION
