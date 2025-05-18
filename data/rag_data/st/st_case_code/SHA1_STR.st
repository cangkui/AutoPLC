FUNCTION_BLOCK SHA1_STR
VAR_INPUT
    RUN : BOOL;
END_VAR

VAR_OUTPUT
    DONE : BOOL;
END_VAR

VAR_IN_OUT
    STR : STRING(OSCAT_BASIC.STRING_LENGTH);
    SHA1 : ARRAY[0..19] OF BYTE;
END_VAR

VAR
    run_last : BOOL;
    SHA1_STREAM : SHA1_STREAM;
    buf : ARRAY[0..63] OF BYTE;
    mode : INT;
    size : UDINT;
    pos : UDINT;
END_VAR
CASE mode OF
0:	(* Wait für Start *)
	IF RUN AND NOT run_last THEN
		DONE := FALSE;
		mode := 1; (* Initialisierung *)
		size := LEN(STR);
	END_IF;

2:	(* Data copy and Calc *)
	IF size > 0 THEN
		OSCAT_BASIC._STRING_TO_BUFFER(STR:=MID(STR, UDINT_TO_INT(SIZE), UDINT_TO_INT(POS) + 1), POS:=0, PT:=ADR(buf), SIZE:=SIZEOF(buf));
	END_IF;

3:	DONE := TRUE;
	mode := 0; (* Stop *)

END_CASE;

IF mode > 0 THEN
	SHA1_STREAM(SIZE:=size, MODE:=mode, BUF:=buf, SHA1:=SHA1, POS=>pos);
END_IF;

run_last := RUN;

(* revision history

ks	6. dec. 2009 rev 1.0
	original version

*)
END_FUNCTION_BLOCK
