FUNCTION_BLOCK ACTUATOR_PUMP
VAR_INPUT
    IN : BOOL;
    MANUAL : BOOL;
    RST : BOOL;
    MIN_ONTIME : TIME := TIME#10s0ms;
    MIN_OFFTIME : TIME := TIME#10s0ms;
    RUN_EVERY : TIME := TIME#10000m0s0ms;
END_VAR

VAR_OUTPUT
    PUMP : BOOL;
END_VAR

VAR_IN_OUT
    RUNTIME : UDINT;
    CYCLES : UDINT;
END_VAR

VAR
    tx : TIME;
    last_change : TIME;
    meter : OSCAT_BASIC.ontime; (* Derived type *)
    old_man : BOOL;
    init : BOOL;
END_VAR
tx := DWORD_TO_TIME(OSCAT_BASIC.T_PLC_MS());

IF NOT init THEN
	init := TRUE;
	last_change := tx;
ELSIF rst THEN
	rst := FALSE;
	runtime := 0;
	cycles := 0;
ELSIF manual AND NOT pump AND NOT old_man THEN
	last_change := tx;
	pump := TRUE;
ELSIF NOT manual AND old_man AND pump AND NOT in THEN
	last_change := tx;
	pump := FALSE;
ELSIF in AND NOT pump AND tx - last_change >= min_offtime THEN
	last_change := tx;
	pump := TRUE;
ELSIF pump AND NOT in AND NOT manual AND tx - last_change >= min_ontime THEN
	last_change := tx;
	pump := FALSE;
ELSIF NOT pump AND (tx - last_change >= run_every) AND (run_every > T#0s) THEN
	last_change := tx;
	pump := TRUE;
END_IF;

meter(in := pump, seconds := runtime, cycles := cycles);
old_man := manual;



(*
hm	27.12.2006		rev 1.1
	fixed a failure while the pump would run for tmin after startup.

hm	15.9.2007		rev 1.2
	replaced Time() with T_PLC_MS for compatibility and performance reasons

hm	13. oct. 2008	rev 1.3
	auto activation can now be disabled when run_every = t#0s

hm	21. oct. 2008	rev 1.4
	changed to use ontime rev 2.0

*)
END_FUNCTION_BLOCK
