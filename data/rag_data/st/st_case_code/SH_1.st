FUNCTION_BLOCK SH_1
VAR_INPUT
    in : REAL;
    PT : TIME;
END_VAR

VAR_OUTPUT
    out : REAL;
    Trig : BOOL;
END_VAR

VAR
    last : TIME;
    tx : TIME;
END_VAR
(* read system time *)
tx := DWORD_TO_TIME(T_PLC_MS());

IF tx - last >= PT THEN
	last := tx;
	out := in;
	trig := TRUE;
ELSE
	trig := FALSE;
END_IF;

(* revision history

HM	6.1.2007	rev 1.1
	added trig output

HM	17.9.2007	rev 1.2
	replaced time() with T_PLC_MS() for compatibility reasons
*)
END_FUNCTION_BLOCK
