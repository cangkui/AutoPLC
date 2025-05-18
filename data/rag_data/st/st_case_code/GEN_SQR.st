FUNCTION_BLOCK GEN_SQR
VAR_INPUT
    PT : TIME; 
    AM : REAL := 1.0; 
    OS : REAL; 
    DC : REAL := 0.5; 
    DL : REAL; 
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
(* check dc = 1 or 0 *)
IF dc = 0.0 THEN
	out := -am * 0.5 + os;
	Q := FALSE;
	RETURN;
ELSIF dc = 1.0 THEN
	out := am * 0.5 + os;
	Q := TRUE;
	RETURN;
END_IF;

(* read system time and prepare input data *)
tx := DWORD_TO_TIME(T_PLC_MS()) - last;
DL := MODR(dl,1.0);
IF dl < 0.0 THEN dl := 1.0 - dl; END_IF;
dc := MODR(dc,1.0);
IF dc < 0.0 THEN dc := 1.0 - dc; END_IF;

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
END_IF;

(* check if falling or rising edge first *)
IF MULTIME(pt, dl + dc) >= pt THEN
	(* generate falling edge *)
	IF tx >= MULTIME(pt, dl + dc - 1) THEN
		out := -am * 0.5 + os;
		Q := FALSE;
	END_IF;
	(* generate rising edge *)
	IF tx >= MULTIME(pt, dl) THEN
		out := am * 0.5 + os;
		Q := TRUE;
	END_IF;
ELSE
	(* generate rising edge first *)
	IF tx >= MULTIME(pt, dl) THEN
		out := am * 0.5 + os;
		Q := TRUE;
	END_IF;
	(* generate falling edge *)
	IF tx >= MULTIME(pt, dl + dc) THEN
		out := -am * 0.5 +os;
		Q := FALSE;
	END_IF;
END_IF;

(* revision history

hm	12. feb 2007	rev 1.1
	added default value for dc = 0.5

hm	17 sep 2007	rev 1.2
	replaced time() with t_plc_ms for compatibilitx reasons

hm	6. jan 2008	rev 1.3
	improved performance

hm	11. mar. 2009	rev 1.4
	changed real constants to use dot syntax
	set default amplitude to 1.0

*)
END_FUNCTION_BLOCK
