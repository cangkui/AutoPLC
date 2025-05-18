FUNCTION_BLOCK BLIND_SHADE_S
VAR_INPUT
    UP : BOOL;
    DN : BOOL;
    S_IN : BYTE;
    PI : BYTE;
    ENABLE : BOOL;
    SUN : BOOL;
    HORZ1 : REAL := 100.0;
    HORZ2 : REAL := 260.0;
    VERT : REAL := 90.0;
    ALERT : BOOL := FALSE;
    sunrise_offset : TIME := TIME#60m0s0ms;
    sunset_preset : TIME := TIME#60m0s0ms;
    shade_delay : TIME := TIME#1m0s0ms;
    shade_pos : BYTE;
END_VAR

VAR_IN_OUT
    CX : OSCAT_BASIC.calendar;
END_VAR_IN_OUT

VAR_OUTPUT
    QU : BOOL;
    QD : BOOL;
    STATUS : BYTE;
    PO : BYTE;
END_VAR_OUTPUT

VAR
    sun_delay : OSCAT_BASIC.TOF;
END_VAR
(* status definition
   151 = shadow               move shutter down for shadowing
   152 = alert                  move shutter up by setting QU:=TRUE when door is open.
*)

(* the input sun is sent through tof which will delay the shade for the time shade_delay *)
sun_delay(IN := sun, PT := shade_delay);

IF ALERT THEN
   QU := TRUE;
   QD := FALSE;
   STATUS := 152;
ELSIF UP AND DN AND ENABLE AND sun_delay.Q
AND (CX.SUN_HOR > HORZ1) AND (CX.SUN_HOR < HORZ2)
AND (CX.SUN_VER < VERT)
AND (DT_TO_TOD(CX.UTC) > CX.SUN_RISE + sunrise_offset) AND (DT_TO_TOD(CX.UTC) < CX.SUN_SET - sunset_preset) THEN
   QU := UP;
   QD := DN;
   STATUS := 151;
   (* Calculate Position, must be shade_pos or less *)
   PO := MIN(PI, shade_pos);
ELSE
   QU := UP;
   QD := DN;
   PO := PI;
   STATUS := S_IN;
END_IF;



(* revision history
hf	12 nov. 2009	rev 1.0
	original version

*)
END_FUNCTION_BLOCK
