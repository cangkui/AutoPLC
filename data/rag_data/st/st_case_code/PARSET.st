FUNCTION_BLOCK PARSET
VAR_INPUT
    A0 : BOOL;
    A1 : BOOL;
    X01 : REAL;
    X02 : REAL;
    X03 : REAL;
    X04 : REAL;
    X11 : REAL;
    X12 : REAL;
    X13 : REAL;
    X14 : REAL;
    X21 : REAL;
    X22 : REAL;
    X23 : REAL;
    X24 : REAL;
    X31 : REAL;
    X32 : REAL;
    X33 : REAL;
    X34 : REAL;
    TC : TIME;
END_VAR

VAR_OUTPUT
    P1 : REAL;
    P2 : REAL;
    P3 : REAL;
    P4 : REAL;
END_VAR

VAR
    X : ARRAY[0..3, 1..4] OF REAL;
    S1 : REAL;
    S2 : REAL;
    S3 : REAL;
    S4 : REAL;
    tx : DWORD;
    last : DWORD;
    start : BOOL;
    set : BYTE;
    init : BOOL;
END_VAR
(* read system_time *)
tx := T_PLC_MS();

(* init sequence *)
IF NOT init THEN
	set.0 := NOT A0;
	init := TRUE;
	X[0,1] := X01;
	X[0,2] := X02;
	X[0,3] := X03;
	X[0,4] := X04;
	X[1,1] := X11;
	X[1,2] := X12;
	X[1,3] := X13;
	X[1,4] := X14;
	X[2,1] := X21;
	X[2,2] := X22;
	X[2,3] := X23;
	X[2,4] := X24;
	X[3,1] := X31;
	X[3,2] := X32;
	X[3,3] := X33;
	X[3,4] := X34;
	P1 := X01;
	P2 := X02;
	P3 := X03;
	P4 := X04;
END_IF;

(* check for input change *)
IF (A0 XOR set.0) OR (A1 XOR set.1) THEN
	(* a new set is selected *)
	set.0 := A0;
	set.1 := A1;
	IF tc > t#0s THEN
		start := TRUE;
		last := tx;
		(* save the step speed for the output changes in S *)
		S1 := (X[set,1] - P1)/DWORD_TO_REAL(TIME_TO_DWORD(tc));
		S2 := (X[set,2] - P2)/DWORD_TO_REAL(TIME_TO_DWORD(tc));
		S3 := (X[set,3] - P3)/DWORD_TO_REAL(TIME_TO_DWORD(tc));
		S4 := (X[set,4] - P4)/DWORD_TO_REAL(TIME_TO_DWORD(tc));
	END_IF;
ELSIF start AND tx - last < TIME_TO_DWORD(tc) THEN
	(* ramp the outputs to the new value *)
	P1 := X[set,1] - S1 * DWORD_TO_REAL(TIME_TO_DWORD(Tc) - tx + last);
	P2 := X[set,2] - S2 * DWORD_TO_REAL(TIME_TO_DWORD(Tc) - tx + last);
	P3 := X[set,3] - S3 * DWORD_TO_REAL(TIME_TO_DWORD(Tc) - tx + last);
	P4 := X[set,4] - S4 * DWORD_TO_REAL(TIME_TO_DWORD(Tc) - tx + last);
ELSE
	(* make sure outputs match the correct set values *)
	start := FALSE;
	P1 := X[set,1];
	P2 := X[set,2];
	P3 := X[set,3];
	P4 := X[set,4];
END_IF;

(* revision history
hm	2.11.2007		rev 1.0
	original version

hm	16. mar. 2008	rev 1.1
	added type conversions to avoid warnings under codesys 3.0

*)
END_FUNCTION_BLOCK
