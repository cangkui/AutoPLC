FUNCTION_BLOCK BLIND_NIGHT
VAR_INPUT
    UP : BOOL;
    DN : BOOL;
    S_IN : BYTE;
    PI : BYTE;
    AI : BYTE;
    E_NIGHT : BOOL := TRUE;
    E_DAY : BOOL := TRUE;
    DTIN : DT;
    SUNRISE : TOD;
    SUNSET : TOD;
    SUNRISE_OFFSET : TIME;
    SUNSET_OFFSET : TIME;
    NIGHT_POSITION : BYTE;
    NIGHT_ANGLE : BYTE;
END_VAR

VAR_OUTPUT
    QU : BOOL;
    QD : BOOL;
    STATUS : BYTE;
    PO : BYTE;
    AO : BYTE;
END_VAR

VAR
    night : BOOL;
    last_night : DATE;
    last_day : DATE;
END_VAR
IF NOT (up AND dn) AND night THEN
	(* manual operation at night will cancel operation for one night *)
	night := FALSE;
ELSIF (DT_TO_TOD(dtin) > sunset + sunset_offset) AND (last_night < DT_TO_DATE(dtin)) AND NOT night AND e_night THEN
	(* enable night *)
	night := TRUE;
	last_night := DT_TO_DATE(dtin);
ELSIF (DT_TO_TOD(dtin) > sunrise + sunrise_offset) AND (last_day < DT_TO_DATE(dtin)) AND night AND e_day AND (last_night < DT_TO_DATE(dtin)) THEN
	(* disable night *)
	night := FALSE;
	last_day := DT_TO_DATE(dtin);
END_IF;

(* shade at night only in auto mode and enable = true *)
IF UP AND DN AND night THEN
	status := 141;
	po := night_position;
	ao := night_angle;
ELSE
	QU := UP;
	QD := DN;
	po := pi;
	ao := ai;
	status := s_in;
END_IF;


(* revision history
hm	29. sep 2007	rev 1.0
	original version

hm	5. oct 2007		rev 1.1
	added enable input

hm	6. oct 2007		rev 1.2
	added pos and angle inputs and outputs
	night position and angle can now be configured
	any manual operation at night will cancel night operation

*)
END_FUNCTION_BLOCK
