FUNCTION_BLOCK BASE64_DECODE_STREAM
VAR_IN_OUT
    BUF1 : ARRAY[0..63] OF BYTE;
    BUF2 : ARRAY[0..47] OF BYTE;
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
    o : INT;
    off : INT;
    i1 : INT;
    i2 : INT;
    c : INT;
END_VAR
IF SIZE1 <= 0 THEN SIZE2 := 0; RETURN; END_IF;
c := SIZE1 - 1;
i2 := 0;
a := 0;
b := 0;
FOR i1 := 0 TO c DO
	o := BUF1[i1];
	IF o = 61  (* = *) THEN	EXIT;
	ELSIF o > 96 (* a-z *) THEN off := 71;
	ELSIF o > 64 (* A-Z *) THEN off := 65;
	ELSIF o > 47 (* 0-9 *) THEN off := -4;
	ELSIF o = 43 (* + *) THEN off := -19;
	ELSIF o = 47 (* / *) THEN off := -16;
	END_IF;
	b := SHL(b,6) OR INT_TO_BYTE(o - off);
	a := a + 6;
	IF a >= 8 THEN
		a := a - 8;
		BUF2[i2] := WORD_TO_BYTE(SHR(b,a));
		i2 := i2 + 1;
		b := b AND NOT SHL(WORD#16#FFFF,a);
	END_IF;
END_FOR;
SIZE2 := i2;

(* revision history

ks	6. dec. 2009 rev 1.0
	original version

*)
END_FUNCTION_BLOCK
