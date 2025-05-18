FUNCTION_BLOCK MANUAL_2
VAR_INPUT
    IN : BOOL;
    ENA : BOOL;
    ON : BOOL;
    OFF : BOOL;
    MAN : BOOL;
END_VAR

VAR_OUTPUT
    Q : BOOL;
    STATUS : BYTE;
END_VAR
IF ena THEN
	IF NOT ON AND NOT OFF THEN
		Q := IN;
		STATUS := 100;
	ELSIF on AND NOT off THEN
		Q := TRUE;
		STATUS := 101;
	ELSIF NOT on AND off THEN
		q := FALSE;
		STATUS := 102;
	ELSE
		Q := MAN;
		STATUS := 103;
	END_IF;
ELSE
	Q := FALSE;
	STATUS := 104;
END_IF;



(* revision history
hm	22. sep. 2008		rev 1.0
	original version

*)
END_FUNCTION_BLOCK
