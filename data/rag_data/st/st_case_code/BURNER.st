FUNCTION_BLOCK BURNER
VAR_INPUT
    in : BOOL;
    stage2 : BOOL;
    over_temp : BOOL;
    oil_temp : BOOL := TRUE;
    Flame : BOOL;
    rst : BOOL;
    rst_timer : BOOL;
    pre_heat_time : TIME := TIME#5s0ms;
    pre_vent_time : TIME := TIME#15s0ms;
    pre_ignite_time : TIME := TIME#5s0ms;
    post_ignite_time : TIME := TIME#25s0ms;
    stage2_delay : TIME := TIME#10s0ms;
    safety_time : TIME := TIME#5s0ms;
    lockout_time : TIME := TIME#10s0ms;
    multiple_ignition : BOOL := TRUE;
    KW1 : REAL;
    KW2 : REAL;
END_VAR

VAR_OUTPUT
    motor : BOOL;
    coil1 : BOOL;
    coil2 : BOOL;
    pre_heat : BOOL;
    ignite : BOOL;
    KWh : REAL;
    status : BYTE;
    fail : BOOL;
END_VAR

VAR_IN_OUT
    runtime1 : UDINT;
    runtime2 : UDINT;
    cycles : UDINT;
END_VAR

VAR
    state : BYTE;
    last : TIME;
    tx : TIME;
    last_change : TIME;
    timer1 : OSCAT_BASIC.ONTIME;
    timer2 : OSCAT_BASIC.ONTIME;
    oil_temp_last : BOOL;
    cycles2 : UDINT;
END_VAR
(* read system time *)
tx := DWORD_TO_TIME(OSCAT_BASIC.T_PLC_MS());

(* check rst input and overtemp *)
IF rst OR over_temp OR state = 0 THEN
	IF status > 0 AND tx - last_change >= lockout_time AND rst THEN
		status := 110;
		fail := FALSE;
		state := 1;
	ELSE
		(* normaler reset *)
		motor := FALSE;
		coil1 := FALSE;
		coil2 := FALSE;
		ignite := FALSE;
		pre_heat := FALSE;
		IF over_temp THEN
			status := 9;
			fail := TRUE;
		END_IF;
		last_change := tx;
		last := tx;
		state := 1;
	END_IF;
END_IF;

(* check for timer rst and rst timer if true *)
IF rst_timer THEN
	runtime1 := 0;
	runtime2 := 0;
	cycles := 0;
	cycles2 := 0;
END_IF;

(* quit here if an error is present *)
IF (status > 0 AND status < 100) OR rst THEN RETURN; END_IF;

(* start sequence *)
CASE state OF

1:	(* in signal starts oil pre heating *)
	IF in AND flame THEN
		state := 7;
		pre_heat := FALSE;
		status := 2;
		last_change := tx;
	ELSIF in THEN
		pre_heat := TRUE;
		state := 2;
		last_change := tx;
	END_IF;

2:	(* after pre_heating time start motor *)
	IF (tx- last_change >= pre_heat_time AND oil_temp) OR (oil_temp AND NOT oil_temp_last) THEN
		motor := TRUE;
		state := 3;
		last_change := tx;
	(* pre_heat_time ist abgelaufen und oil_temp ist nicht aktiv *)
	ELSIF tx - last_change >= pre_heat_time AND NOT oil_temp THEN
		state := 7;
		pre_heat := FALSE;
		status := 1;
		last_change := tx;
	(* flame monitor cannot be active at this time *)
	ELSIF flame THEN
		state := 7;
		pre_heat := FALSE;
		status := 2;
		last_change := tx;
	END_IF;

3:	(* abwarten bis zündung eingeschaltet werden kann *)
	IF tx - last_change >= pre_vent_time - pre_ignite_time THEN
		ignite := TRUE;
		state := 4;
		last_change := tx;
	(* flame monitor cannot be active at this time *)
	ELSIF flame THEN
		state := 7;
		pre_heat := FALSE;
		motor := FALSE;
		status := 3;
		last_change := tx;
	END_IF;

4:	(* warten bis oelzufuhr geoeffnet werden darf *)
	IF tx - last_change >= pre_ignite_time THEN
		coil1 := TRUE;
		state := 5;
		last_change := tx;
	END_IF;

5:	(* warten auf flammwaechter und falls noetig abschalten *)
	IF tx - last_change >= safety_time OR flame THEN
		IF NOT flame THEN
			(* notabschaltung da flammwaechster nicht angesprochen hat *)
			state := 7;
			motor := FALSE;
			coil1 := FALSE;
			pre_heat := FALSE;
			ignite := FALSE;
			status := 4;
			last_change := tx;
		ELSE
			state := 6;
			last_change := tx;
		END_IF;
	END_IF;

6:	(* brenner läuft, flammueberwachung und nach ablauf der nachigniteszeit ignite abschalten *)
	IF NOT flame AND NOT multiple_ignition THEN
		(* notabschaltung da flammwaechster keine flamme meldet *)
		state := 7;
		motor := FALSE;
		coil1 := FALSE;
		coil2 := FALSE;
		pre_heat := FALSE;
		ignite := FALSE;
		status := 5;
		last_change := tx;
	ELSIF NOT flame AND multiple_ignition THEN
		ignite := TRUE;
		state := 5;
		coil2 := FALSE;
		last_change := tx;
	ELSE
		IF tx - last_change >= post_ignite_time THEN
			(* post_ignite_time abgelaufen, ignite abschalten *)
			ignite := FALSE;
		END_IF;
		IF tx - last_change >= stage2_delay AND stage2 THEN
			coil2 := TRUE;
		ELSE
			coil2 := FALSE;
		END_IF;
	END_IF;
END_CASE;

(* abschaltung wenn kein eingangssignal *)
IF NOT in THEN
	state := 1;
	motor := FALSE;
	coil1 := FALSE;
	coil2 := FALSE;
	ignite := FALSE;
	pre_heat := FALSE;
	last_change := tx;
END_IF;

(* runtimezähler *)
timer1(in := flame AND in AND motor AND coil1 AND NOT coil2, seconds := runtime1, cycles := cycles);
timer2(in := flame AND in AND motor AND coil1 AND coil2, seconds := runtime2, cycles := cycles2);
KWH := UDINT_TO_REAL(runtime1) * KW1 / 3600.0 + UDINT_TO_REAL(runtime2) * KW2 / 3600.0;

(* zeit fuer naechsten aufruf merken *)
last := tx;

(* set fail output IF ERROR and Status  if normal operation *)

IF status > 0 AND status < 100 THEN
	fail := TRUE;
ELSE
	fail := FALSE;
	IF NOT in THEN
		status := 110;
	ELSIF flame AND in AND motor AND coil2 AND coil1 THEN
		status := 113;
	ELSIF flame AND in AND motor AND coil1 THEN
		status := 112;
	ELSE
		status := 111;
	END_IF;
END_IF;



(*
31.12.2006	hm		rev 1.1
	added pre_heat timing and output
	flame monitor before ignition will generate error on status output
	added output for KWh

31.12.2006	hm		rev 1.2
	added oil_temp_input

31.12.2006	hm		rev 1.3
	changed status output to be byte instead of bool

22.1.2007		hm		rev 1.4
	renamed fail output to status
	added output cycles
	corrected an error while pre_heat would be on when an error was present
	added coil2 output and stage2 input

15.9.2007		hm		rev 1.5
	replaced Time() with T_PLC_MS for compatibility and performance reasons

hm	21. oct. 2008	rev 1.6
	changed code to use new ontime rev 2.0	
*)
END_FUNCTION_BLOCK
