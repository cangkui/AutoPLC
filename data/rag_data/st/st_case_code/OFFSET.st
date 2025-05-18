FUNCTION OFFSET : REAL
VAR_INPUT
    X : REAL;
    O1 : BOOL;
    O2 : BOOL;
    O3 : BOOL;
    O4 : BOOL;
    D : BOOL;
    Offset_1 : REAL;
    Offset_2 : REAL;
    Offset_3 : REAL;
    Offset_4 : REAL;
    default : REAL;
END_VAR
IF D THEN OFFSET := default; ELSE OFFSET := X; END_IF;
IF O1 THEN OFFSET := OFFSET + offset_1; END_IF;
IF O2 THEN OFFSET := OFFSET + offset_2; END_IF;
IF O3 THEN OFFSET := OFFSET + offset_3; END_IF;
IF O4 THEN OFFSET := OFFSET + offset_4; END_IF;



(* revision history
hm	12. jan 2007
	original version


*)
END_FUNCTION
