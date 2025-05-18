FUNCTION_BLOCK TP_1D
VAR_INPUT
    IN : BOOL;
    PT1 : TIME;
    PTD : TIME;
    RST : BOOL;
END_VAR

VAR_OUTPUT
    Q : BOOL;
    W : BOOL;
END_VAR

VAR
    tx : TIME;
    start : TIME;
    ix : BOOL;
END_VAR
(* read system timer *)
tx := DWORD_TO_TIME(T_PLC_MS());

IF RST THEN
	Q := FALSE;
	rst := FALSE;
	W := FALSE;
ELSIF W THEN
	IF tx - start >= PTD THEN
		W := FALSE;
	END_IF;
ELSIF IN AND NOT ix THEN
	Q := TRUE;
	start := tx;
	in := FALSE;
ELSIF tx - start >= PT1 THEN
	Q := FALSE;
	W := TRUE;
	start := tx;
END_IF;

ix := IN;



(* revision history
hm	28. jun. 2008
	original version

*)
END_FUNCTION_BLOCK
