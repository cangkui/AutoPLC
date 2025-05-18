FUNCTION_BLOCK LIST_NEXT
VAR_INPUT
    SEP : BYTE;
    RST : BOOL;
END_VAR

VAR_IN_OUT
    LIST : STRING(LIST_LENGTH);
END_VAR_IN_OUT

VAR_OUTPUT
    LEL : STRING(LIST_LENGTH);
    NUL : BOOL;
END_VAR_OUTPUT

VAR
    pos : INT := 1;
    pt : POINTER TO ARRAY[1..LIST_LENGTH] OF BYTE;
    po : POINTER TO ARRAY[1..LIST_LENGTH] OF BYTE;
    c : BYTE;
    write : INT;
END_VAR
pt := ADR(LIST);
po := ADR(LEL);

IF rst THEN
	pos := 1;
END_IF;

IF pt^[pos] = 0 OR pos = LIST_LENGTH THEN
	LEL := '';
	NUL := TRUE;
ELSE
	NUL := FALSE;
	write := 1;
	FOR pos := pos + 1 TO LIST_LENGTH DO
		c := pt^[pos];
		IF c = 0 OR c = SEP THEN
			po^[write] := 0;
			RETURN;
		ELSE
			po^[write] := pt^[pos];
			write := write + 1;
		END_IF;
	END_FOR;
END_IF;




(* revision histroy
hm	25. jun. 2008	rev 1.0
	original release

hm	19. jan. 2011	rev 1.1
	changed string(255) to string(LIST_LENGTH)	

hm	21. mar. 2011	rev 2.0
	all elements start with SEP

*)
END_FUNCTION_BLOCK
