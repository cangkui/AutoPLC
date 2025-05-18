FUNCTION_BLOCK ACTUATOR_COIL
VAR_INPUT
    IN : BOOL;
    self_act_cycle : TIME := TIME#14400m0s0ms;
    self_act_time : TIME := TIME#1s0ms;
END_VAR

VAR_OUTPUT
    OUT : BOOL;
    STATUS : BYTE;
END_VAR

VAR
    last : DWORD;
    init : BOOL;
    tx : DWORD;
    now : DWORD;
END_VAR
(* read system time *)
now := OSCAT_BASIC.T_PLC_MS();

(* initialize for first cycle *)
IF NOT init THEN
	last := now;
	init := TRUE;
ELSIF IN THEN
	OUT := TRUE;
	STATUS := 101; (* activated by input *)
	LAST := now;
ELSE
	OUT := FALSE;
	STATUS := 100; (* disabled *)
	(* now we need to check for self activation *)
	tx := now - last;
	IF (self_act_cycle > T#0s) AND (tx >= TIME_TO_DWORD(self_act_cycle)) THEN
		OUT := TRUE;
		STATUS := 102; (* auto activation *)
		IF tx >= TIME_TO_DWORD(self_act_cycle + self_act_time) THEN
			last := now;
			OUT := FALSE;
			STATUS := 100; (* idle *)
		END_IF;
	END_IF;
END_IF;



(*
revision history:

hm		1. jun. 2008 	rev 1.0
	original version

*)
END_FUNCTION_BLOCK
