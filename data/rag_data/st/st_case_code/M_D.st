FUNCTION_BLOCK M_D
VAR_INPUT
    start : BOOL;
    stop : BOOL;
    tmax : TIME := TIME#14400m0s0ms;
    rst : BOOL;
END_VAR

VAR_OUTPUT
    PT : TIME;
    ET : TIME;
    run : BOOL;
END_VAR

VAR
    edge : BOOL;
    T0 : TIME;
    tx : TIME;
    startup : BOOL;
END_VAR
(* check for rst input *)
IF rst OR et >= tmax THEN
	pt := t#0ms;
	et := t#0ms;
	startup := FALSE;
	run := FALSE;
END_IF;

(* avoid timers to start when input is true at startup *)
IF NOT startup THEN
	edge := start;
	startup := TRUE;
END_IF;

(* read system timer *)
tx := DWORD_TO_TIME(T_PLC_MS());

(* detect rising edge on start *)
IF start AND NOT edge AND NOT stop THEN
	t0 := tx;
	run := TRUE;
	pt := t#0s;
ELSIF stop AND run THEN
	pt := et;
	run := FALSE;
END_IF;
edge := start;
IF run THEN et := tx - t0; END_IF;

(* revision history
hm		2.5.2007	rev 1.0
	original version

hm		16.9.2007	rev 1.1
	changes time() to T_plc_ms() for compatibility reasons

hm	27. feb 2009	rev 1.2
	deleted unnecessary init with 0
*)
END_FUNCTION_BLOCK
