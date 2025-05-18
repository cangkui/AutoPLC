FUNCTION_BLOCK ACTUATOR_2P
VAR_INPUT
    IN : BYTE;
    TEST : BOOL;
    ARE : BOOL := TRUE;
    CYCLE_TIME : TIME;
    SENS : BYTE;
    SELF_ACT_TIME : TIME;
    SELF_ACT_PULSE : TIME;
    SELF_ACT_CYCLES : INT := 1;
END_VAR

VAR_IN_OUT
    ARX : BOOL;
END_VAR

VAR_OUTPUT
    OUT : BOOL;
    ARO : BOOL;
END_VAR

VAR
    timer : AUTORUN;
    pwgen : OSCAT_BASIC.GEN_PULSE;
END_VAR
(* run the autorun timer *)
timer(trun := SELF_ACT_PULSE * (2 * SELF_ACT_CYCLES), toff := SELF_ACT_TIME, Test := TEST, ARE := ARE, arx := ARX);
ARO := timer.ARO;

(* run the pulse width generator *)
IF aro THEN
	pwgen(PTL := SELF_ACT_PULSE, PTH := SELF_ACT_PULSE);
	out := pwgen.Q;
ELSIF in < sens THEN
	out := FALSE;
ELSIF in > BYTE#255 - sens THEN
	out := TRUE;
ELSE
	pwgen.PTH := (CYCLE_TIME * OSCAT_BASIC.BAND_B(IN,SENS)) / 255;
	pwgen.PTL := CYCLE_TIME - pwgen.PTH;
	pwgen();
	out := pwgen.Q;
END_IF;



(* revision history
hm	7.10.2006 		rev 1.1
	changed error pos could not reach 1 or 0 under certain conditions due to error in code.
	changed on and off in force and force_in for better usability.

hm	17.1.2007		rev 1.2
	deleted unused variable time_overflow

hm	15.9.2007		rev 1.3
	replaced time() with T_PLC_MS for compatibility and performance reasons

hm	19. 11 2007		rev 1.4
	replaced left over statement time() with tx

hm	27. 12. 2007	REV 1.5
	CHANGED CODE FOR BETTER PERFORMANCE

hm	21. oct. 2008	rev 1.6
	code optimized

hm	23. nov. 2008	rev 2.0
	new code using library modules

hm	24. jan. 2009	rev 2.1
	deleted unused var inb
*)
END_FUNCTION_BLOCK
