FUNCTION_BLOCK TMAX
VAR_INPUT
    IN     : BOOL; 
    PT     : TIME; 
END_VAR

VAR_OUTPUT
    Q      : BOOL; 
    Z      : BOOL; 
END_VAR

VAR
    tx      : TIME; 
    start    : TIME; 
    last_in  : BOOL; 
END_VAR
(* read system timer *)
tx := DWORD_TO_TIME(T_PLC_MS());

Z := FALSE;

IF NOT in THEN
	Q := FALSE;
ELSIF IN AND NOT last_in THEN
	Q := TRUE;
	start := tx;
ELSIF (tx - start >= PT) AND Q THEN
	Q := FALSE;
	Z := TRUE;
END_IF;

last_in := IN;



(* revision history
hm	20. jul. 2008	rev 1.0
	original version

*)
END_FUNCTION_BLOCK
