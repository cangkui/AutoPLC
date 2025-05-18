FUNCTION_BLOCK GEN_RDM
VAR_INPUT
    PT : TIME;
    AM : REAL := 1;
    OS : REAL;
END_VAR

VAR_OUTPUT
    Q : BOOL;
    Out : REAL;
END_VAR

VAR
    tx : TIME;
    last : TIME;
    init : BOOL;
END_VAR
(* read system time and prepare input data *)
tx := DWORD_TO_TIME(T_PLC_MS()) - last;

(* init section *)
IF NOT init THEN
	init := TRUE;
	last := tx;
	tx := t#0s;
END_IF;

(* add last if one cycle is finished *)
IF tx >= pt THEN
	last := last + pt;
	tx := tx - pt;

	(* generate output signal *)
	out := am * (RDM(0) - 0.5) + os;
	q := TRUE;
ELSE
	q := FALSE;
END_IF;

(* revision history

hm	7.2.2007		rev 1.0
	original version

hm	16.9.2007		rev 1.1
	changes time() to T_plc_ms() for compatibility reasons

*)
END_FUNCTION_BLOCK
