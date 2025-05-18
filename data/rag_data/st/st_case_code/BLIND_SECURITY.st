FUNCTION_BLOCK BLIND_SECURITY
VAR_INPUT
    UP : BOOL;
    DN : BOOL;
    S_IN : BYTE;
    PI : BYTE;
    AI : BYTE;
    FIRE : BOOL;
    WIND : BOOL;
    ALARM : BOOL;
    DOOR : BOOL;
    RAIN : BOOL;
    Alarm_UP : BOOL := TRUE;
    WIND_UP : BOOL := TRUE;
    RAIN_UP : BOOL := FALSE;
END_VAR

VAR_OUTPUT
    QU : BOOL;
    QD : BOOL;
    STATUS : BYTE;
    PO : BYTE;
    AO : BYTE;
END_VAR
IF Fire THEN
	QU := TRUE;
	QD := FALSE;
	status := 111;
ELSIF Wind THEN
	QU := Wind_up;
	QD := NOT wind_up;
	status := 112;
ELSIF alarm THEN
	QU := Alarm_up;
	QD := NOT Alarm_up;
	status := 113;
ELSIF Door THEN
	QU := TRUE;
	QD := FALSE;
	status := 114;
ELSIF Rain AND NOT (up XOR dn) THEN
	QU := rain_up;
	QD := NOT rain_up;
	status := 115;
ELSE
	QU := UP;
	QD := DN;
	status := S_IN;
	po := pi;
	ao := ai;
END_IF;


(* revision history
hm	29. sep 2007	rev 1.0
	original release

hm	6. oct 2007		rev 1.1
	added pos and angle inputs and outputs
	manual will override rain position
	alarm position can now be configured

hm		12. feb. 2008	rev 1.2
	added config variables rain_up and wind_up
	position for rain and wind is now configurable
*)
END_FUNCTION_BLOCK
