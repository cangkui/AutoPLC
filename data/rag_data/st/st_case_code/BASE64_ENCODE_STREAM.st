FUNCTION_BLOCK BASE64_ENCODE_STREAM
VAR_IN_OUT
    BUF1 : ARRAY[0..47] OF BYTE;
    BUF2 : ARRAY[0..63] OF BYTE;
END_VAR

VAR_INPUT
    SIZE1 : INT;
END_VAR

VAR_OUTPUT
    SIZE2 : INT;
END_VAR

VAR
    a : INT;
    b : WORD;
    i : INT;
    i2 : INT;
    i3 : INT;
    c : INT;
    BASE64 : STRING(64) := 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
END_VAR
IF SIZE1 <= 0 THEN
	SIZE2 := 0;
	RETURN;
END_IF;
c := SIZE1 - 1;
i2 := 0;
a := 0;
b := 0;
FOR i := 0 TO c DO
	b := SHL(b,8) OR BUF1[i];
	a := a + 8;
	WHILE a >= 6 DO
		a := a - 6;
		buf2[i2] := OSCAT_BASIC.CODE(BASE64,WORD_TO_INT(SHR(b,a)) + 1);
		i2 := i2 + 1;
		b := b AND NOT SHL(WORD#16#FFFF,a);
	END_WHILE;
END_FOR;
IF a > 0 THEN
	buf2[i2] := OSCAT_BASIC.CODE(BASE64,WORD_TO_INT(SHL(b,6 - a)) + 1);
	i2 := i2 + 1;
	i3 := i2 + (NOT INT_TO_WORD(i2) AND 2#11);
	FOR i := i2 TO i3 DO
		buf2[i] := 61;
	END_FOR;
	SIZE2 := i3 + 1;
ELSE
	SIZE2 := i2;
END_IF;

(* revision history

ks	6. dec. 2009 rev 1.0
	original version

*)
END_FUNCTION_BLOCK
