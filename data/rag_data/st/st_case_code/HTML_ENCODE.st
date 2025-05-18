FUNCTION HTML_ENCODE : string
VAR_INPUT
    IN : STRING(OSCAT_BASIC.STRING_LENGTH); 
    M : BOOL; 
END_VAR

VAR
    pt_in : POINTER TO ARRAY[1..255] OF BYTE; 
    pt_out : POINTER TO ARRAY[1..256] OF BYTE; 
    pos_in : INT; 
    pos_out : INT; 
    stop : INT; 
    B : BYTE; 
    tmp : STRING(10); 
    i : INT; 
END_VAR
pt_in  := ADR(in);
pt_out := ADR(HTML_ENCODE);

(* scan the input string and convert unallowed characters *)
pos_in := 1;
stop := LEN(in);
FOR pos_out := 1 TO 250 DO
	IF pos_in > stop THEN EXIT; END_IF;
	B := pt_in^[pos_in];
	CASE BYTE_TO_INT(B) OF
		34: (* " detected *)
			IF pos_out > OSCAT_BASIC.STRING_LENGTH - 6 THEN EXIT; END_IF;
			pt_out^[pos_out] := 38;
			pos_out := pos_out + 1;
			pt_out^[pos_out] := 113;
			pos_out := pos_out + 1;
			pt_out^[pos_out] := 117;
			pos_out := pos_out + 1;
			pt_out^[pos_out] := 111;
			pos_out := pos_out + 1;
			pt_out^[pos_out] := 116;
			pos_out := pos_out + 1;
			pt_out^[pos_out] := 59;
		38: (* & detected *)
			IF pos_out > OSCAT_BASIC.STRING_LENGTH - 5 THEN EXIT; END_IF;
			pt_out^[pos_out] := 38;
			pos_out := pos_out + 1;
			pt_out^[pos_out] := 97;
			pos_out := pos_out + 1;
			pt_out^[pos_out] := 109;
			pos_out := pos_out + 1;
			pt_out^[pos_out] := 112;
			pos_out := pos_out + 1;
			pt_out^[pos_out] := 59;
		60: (* < detected *)
			IF pos_out > OSCAT_BASIC.STRING_LENGTH - 4 THEN EXIT; END_IF;
			pt_out^[pos_out] := 38;
			pos_out := pos_out + 1;
			pt_out^[pos_out] := 108;
			pos_out := pos_out + 1;
			pt_out^[pos_out] := 116;
			pos_out := pos_out + 1;
			pt_out^[pos_out] := 59;
		62: (* > detected *)
			IF pos_out > OSCAT_BASIC.STRING_LENGTH - 4 THEN EXIT; END_IF;
			pt_out^[pos_out] := 38;
			pos_out := pos_out + 1;
			pt_out^[pos_out] := 103;
			pos_out := pos_out + 1;
			pt_out^[pos_out] := 116;
			pos_out := pos_out + 1;
			pt_out^[pos_out] := 59;
	ELSE
		IF M AND B > 127 THEN
			tmp := OSCAT_BASIC.CHARNAME(B);
			(* exit if output string has not enough space *)
			IF pos_out + LEN(tmp) + 2 <= 250 THEN
				pt_out^[pos_out] := 38;
				pos_out := pos_out + 1;
				FOR i := 1 TO LEN(tmp) DO
					pt_out^[pos_out] := OSCAT_BASIC.CODE(tmp, i);
					pos_out := pos_out + 1;
				END_FOR;
				pt_out^[pos_out] := 59;
			END_IF;
		ELSE
			pt_out^[pos_out] := B;
		END_IF;
	END_CASE;
	(* increment in by 1 *)
	pos_in := pos_in + 1;
END_FOR;

(* terminate the output string *)
pt_out^[pos_out] := 0;



(* revision history
hm		3. mai. 2008	rev 1.0
	original version

hm		10. mai. 2008	rev 1.1
	stringlength 250
*)
END_FUNCTION
