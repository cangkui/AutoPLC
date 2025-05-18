FUNCTION_BLOCK MYSQL_AUTH
VAR_IN_OUT
    RUN : BOOL;
    PASSWORD : STRING(64);
    MESSAGE : ARRAY[0..19] OF BYTE;
    SCRAMBLE : ARRAY[0..19] OF BYTE;
END_VAR

VAR
    SHA1_STREAM : SHA1_STREAM;
    buf : ARRAY[0..63] OF BYTE;
    sha_mode : INT;
    step : INT;
    size : UDINT;
    pos : UDINT;
    i : INT;
    i2 : INT;
    scramble1 : ARRAY[0..19] OF BYTE;
    scramble2 : ARRAY[0..19] OF BYTE;
END_VAR
IF NOT RUN THEN RETURN; END_IF;
CASE step OF

0:	CASE sha_mode OF
	0:	sha_mode := 1; (* Initialisierung *)
		size := INT_TO_UDINT(LEN(PASSWORD));
		i2 := UDINT_TO_INT(size) - 1;
		FOR i := 0 TO 63 DO
			IF i <= i2 THEN
				buf[i] := OSCAT_BASIC.CODE(STR:=PASSWORD,POS:=i+1);
			ELSE
				buf[i] := BYTE#0;
			END_IF;
		END_FOR;

	3:	scramble1 := SCRAMBLE;
		sha_mode := 1; (* Initialisierung *)
		size := UDINT#20;
		FOR i := 0 TO 19 DO
			buf[i] := scramble1[i];
		END_FOR;
		step := 10;
	END_CASE;

10:	IF sha_mode = 3 THEN
		sha_mode := 1; (* Initialisierung *)
		scramble2 := SCRAMBLE;
		size := UDINT#40;
		FOR i := 0 TO 19 DO
			buf[i] := message[i];
			buf[i+20] := scramble2[i];
		END_FOR;
		step :=20;
	END_IF;

20:	IF sha_mode = 3 THEN
		sha_mode := 0; (* Stop *)
		FOR i:= 0 TO 19 DO
			SCRAMBLE[i] := SCRAMBLE[i] XOR scramble1[i];
		END_FOR;
		step :=0;
		RUN := FALSE;
	END_IF;
END_CASE;

SHA1_STREAM(SIZE:=size, MODE:=sha_mode, BUF:=buf, SHA1:=SCRAMBLE, POS=>pos);

(* revision history
ks	20. apr. 2012	rev 1.0
	original version

*)
END_FUNCTION_BLOCK
