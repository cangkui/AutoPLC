FUNCTION_BLOCK BLIND_SHADE
VAR_INPUT
    UP : BOOL;
    DN : BOOL;
    S_IN : BYTE;
    PI : BYTE;
    AI : BYTE;
    ENABLE : BOOL;
    SUN : BOOL;
    sunrise_offset : TIME := TIME#60m0s0ms;
    sunset_preset : TIME := TIME#60m0s0ms;
    direction : REAL := 180.0;
    angle_offset : REAL := 80.0;
    slat_width : REAL := 80.0;
    Slat_spacing : REAL := 60.0;
    Shade_delay : TIME := TIME#1m0s0ms;
    Shade_pos : BYTE;
END_VAR

VAR_IN_OUT
    CX : OSCAT_BASIC.CALENDAR;
END_VAR

VAR_OUTPUT
    QU : BOOL;
    QD : BOOL;
    STATUS : BYTE;
    PO : BYTE;
    AO : BYTE;
END_VAR

VAR
    angle : REAL;
    sun_delay : OSCAT_BASIC.TOF;
END_VAR
(* the input sun is sent through tof which will delay the shade for the time shade_delay *)
sun_delay(IN := sun, PT := shade_delay);

IF UP AND DN AND enable AND sun_delay.Q AND cx.SUN_HOR > direction - angle_offset AND cx.SUN_HOR < direction + angle_offset AND
	DT_TO_TOD(cx.UTC) > cx.SUN_RISE + sunrise_offset AND DT_TO_TOD(cx.UTC) < cx.SUN_SET - sunset_preset THEN
	status := 151;
	QU := UP;
	QD := DN;
	(* position is predefined *)
	po := shade_pos;
	(* shading is active now calculate the slat angle *)
	(* calculate the max angle for the blind *)
	angle := OSCAT_BASIC.DEG(ATAN(slat_spacing / slat_width));
	(* check if sun angle is between 0 and max angle *)
	IF cx.SUN_VER > 0.0 AND cx.SUN_VER < angle THEN
		angle := cx.SUN_VER + OSCAT_BASIC.DEG(ACOS(COS(OSCAT_BASIC.RAD(cx.SUN_VER))*slat_spacing / Slat_width));
		ao := INT_TO_BYTE(DINT_TO_INT(LIMIT(0,TRUNC(angle * 2.833333333), 255)));
	ELSE
		ao := 255;
	END_IF;
ELSE
	QU := UP;
	QD := DN;
	po := pi;
	ao := ai;
	status := S_IN;
END_IF;

(* revision history
hm	15 oct. 2007	rev 1.0
	original version

hm	19. oct 2007	rev 1.1
	manual mode was not detected because outputs were checked and not inputs UP and DN

hm	24. nov 2007	rev 1.2
	added shade_delay to avoid constant up and down if sun is out parcially

hm	8. feb 2009	rev 1.3
	added shading position shade_pos to be configured
	changed inputs to utilize calendar data structure

hm	13. mar. 2009	rev 1.4
	improved code

*)
END_FUNCTION_BLOCK
