FUNCTION_BLOCK ACTUATOR_UD
VAR_INPUT
    UD : BOOL;
    ON : BOOL;
    Manual : BOOL;
    Up : BOOL;
    Dn : BOOL;
    off : BOOL;
    YUP_in : BOOL;
    YDN_in : BOOL;
    TON : TIME;
    TOFF : TIME;
    out_return : BOOL;
END_VAR

VAR_OUTPUT
    Yup : BOOL;
    Ydn : BOOL;
    status : BYTE;
END_VAR

VAR
    tx : TIME;
    last : TIME;
    init : BOOL;
END_VAR
(* read system time *)
tx := DWORD_TO_TIME(OSCAT_BASIC.T_PLC_MS());

IF NOT init THEN
	(* INITIALIZATION AT FIRST CALL *)
	last := tx;
	init := TRUE;
ELSIF off THEN
	(* emergency shut off detected *)
	Yup := FALSE;
	ydn := FALSE;
	last := tx;
	status := 101;

(* Watch FOR MIN TON TIME *)
ELSIF ((Yup OR Ydn) AND tx - last < Ton) THEN
	RETURN;
ELSIF NOT yup AND NOT ydn AND tx - last < toff THEN
	RETURN;

(* manual operation *)
ELSIF manual THEN
	status := 102;
	(* directional chage set both outs 0 first *)
	IF yup AND NOT up OR ydn AND NOT dn THEN
		ydn := FALSE;
		yup := FALSE;
		last := tx;
	(* manual up operation *)
	ELSIF UP AND NOT DN AND NOT OFF THEN
		Ydn := FALSE;
		Yup := TRUE;
		last := tx;
		status := 103;
	(* manual down operation *)
	ELSIF DN AND NOT UP AND NOT OFF THEN
		Yup := FALSE;
		Ydn := TRUE;
		last := tx;
		status := 104;
	(* maual mode but no up or down *)
	ELSE
		IF YUP OR YDN THEN LAST := tx; END_IF;
		Yup := FALSE;
		ydn := FALSE;
	END_IF;

(* automatic operation *)
ELSE
	(* directional change set outputs low first *)
	IF yup AND NOT ud OR ydn AND UD THEN
		yup := FALSE;
		ydn := FALSE;
		last := tx;
	(* automatic up operation *)
	ELSIF Ud AND on AND NOT OFF THEN
		Ydn := FALSE;
		Yup := TRUE;
		last := tx;
		status := 111;

	(* automatic down operation *)
	ELSIF NOT UD AND on AND NOT OFF THEN
		Yup := FALSE;
		Ydn := TRUE;
		last := tx;
		status := 112;
	(* standby mode *)
	ELSE
		IF Yup OR Ydn THEN last := tx; END_IF;
		Yup := FALSE;
		Ydn := FALSE;
		status := 110;
	END_IF;
END_IF;

(* make sure Yup and Ydn are never on at the same time *)
IF Ydn AND Yup_in AND out_return THEN
	Ydn := FALSE;
	status := 1;
END_IF;
IF Yup AND Ydn_in AND out_return THEN
	Yup := FALSE;
	status := 2;
END_IF;



(* revision history
hm 22. jan 2007	rev 1.1
	added status output
	changed inputs ton, toff to config variables
	added config var out_return to enable yup_in and Ydn_in
	manual mod will override automatic mode 

hm	15.9.2007	rev 1.2
	replaced Time() with T_PLC_MS for compatibility and performance reasons

hm	17. oct. 2008	rev 1.3
	deleted unnecessary initialization with 0
*)
END_FUNCTION_BLOCK
