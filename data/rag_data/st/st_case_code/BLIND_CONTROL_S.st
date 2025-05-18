FUNCTION_BLOCK BLIND_CONTROL_S
VAR_INPUT
    UP : BOOL;
    DN : BOOL;
    S_IN : BYTE := 125; // Initial value
    PI : BYTE;
    T_UP : TIME;
    T_DN : TIME;
    RU : BOOL;
    RD : BOOL;
    T_LOCKOUT : TIME := TIME#100ms; // Initial value
    T_EXT : TIME;
    EXT_TRIG : BYTE := 5; // Initial value
    R_POS_TOP : BYTE := 255; // Initial value
    R_POS_BOT : BYTE;
END_VAR

VAR_OUTPUT
    POS : BYTE;
    MU : BOOL;
    MD : BOOL;
    STATUS : BYTE;
END_VAR

VAR
    rmp : OSCAT_BASIC._RMP_NEXT;
    tx : TIME;
    last : TIME;
    PI_last : BYTE;
END_VAR
(* status definition
	121 = manual up				manual up operation UP = true and DN = false
	122 = manual down			manual down operation DN = true and UP = false
	123 = auto positioning		automatic mode DN = UP = TRUE
	124 = revert					revert from top or bottom
	125 = nothing					default input for S_IN
	127 = Lockout time			lockout between directional change
	128 = calibrate				calibrate after power up
	129 = extend					extend runtime at top or bottom for continuous calibration
*)

(* read system time *)
tx := DWORD_TO_TIME(OSCAT_BASIC.T_PLC_MS());

(* check inputs and start state machine *)
IF UP  AND NOT DN THEN
	(*  manual UP *)
	rmp.IN := 255;
	STATUS := 121;
ELSIF DN AND NOT UP THEN
	(* manual DN *)
	rmp.IN := 0;
	STATUS := 122;
ELSIF NOT (UP OR DN) THEN
	(* manual standby mode *)
	rmp.IN := PI;
	STATUS := S_IN;
END_IF;

(* simulate pos output *)
rmp(E := UP OR DN, TR := T_UP, TF := T_DN, TL := T_LOCKOUT, OUT := POS);

(* state machine *)
CASE STATUS OF
	0:		(* power up init *)
		last := tx;
		PI_last := PI XOR BYTE#255;
		STATUS := 128;		(* calibrate *)

	121: 	(* manual up *)
		MU := TRUE;
		MD := FALSE;
		IF POS >= (BYTE#255 - EXT_TRIG) THEN
			(* extend cycle if top is reached *)
			POS := 255;
			last := tx;
			STATUS := 129;	(* extend *)
		END_IF;

	122:	(* manual down *)
		MD := TRUE;
		MU := FALSE;
		IF POS <= EXT_TRIG THEN
			(* extend cycle if bottom is reached *)
			POS := 0;
			last := tx;
			STATUS := 129;	(* extend *)
		END_IF;

	123:	(* auto positioning *)
		MD := rmp.DN;
		MU := rmp.UP;
		IF NOT (rmp.DN OR rmp.UP) THEN
			(* automatic position is reached *)
			IF POS <= EXT_TRIG THEN
				(* extend at bottom *)
				MD := TRUE;
				last := tx;
				STATUS := 129;	(* extend *)
			ELSIF POS >= (BYTE#255 - EXT_TRIG)  THEN
				(* extend at top *)
				MU := TRUE;
				last := tx;
				STATUS := 129;	(* extend *)
			ELSE
				STATUS := S_IN;
			END_IF;
		END_IF;

	124:	(* revert from top or bottom *)
		MD := rmp.DN;
		MU := rmp.UP;
		IF NOT(rmp.DN OR rmp.UP) THEN
(*		IF POS = rmp.IN THEN  *)
			PI_last := PI;
			STATUS := S_IN;
		END_IF;

	127:	(* lockout time *)
		IF (tx - last) >= T_LOCKOUT THEN
				STATUS := S_IN;
		END_IF;

	128:	(* calibration *)
		MU := TRUE;
		MD := FALSE;
		rmp.IN := 255;
		IF (tx - last) >= (T_UP +  T_EXT) THEN
			MU := FALSE;
			last := tx;
			STATUS := 127;	(* lockout *)
		END_IF;

	129: (* extend mode *)
		IF (tx - last) >= T_EXT THEN
			MU := FALSE;
			MD := FALSE;
			last := tx;
			STATUS := 127;	(* lockout *)
		END_IF;

ELSE
	MU := FALSE;
	MD := FALSE;
	IF PI <> PI_last THEN
		PI_last := PI;
		rmp.IN := PI;
		STATUS := 123;	(* auto positioning *)
	ELSIF POS = 0 AND RU THEN
		rmp.IN := R_POS_BOT;
		STATUS := 124;	(* revert *)
	ELSIF pos = 255 AND RD THEN
		rmp.IN := R_POS_TOP;
		STATUS := 124;	(* revert *)
	ELSE
		STATUS := S_IN;
	END_IF;
END_CASE;



(* revision history

hm	9. mar. 2009	rev 1.0
	original release

hf	27. jul 2009	rev 1.1
	major code changes and updates

hf	23. Jan 2010	rev 1.2
	fixed a bug during calibration
	added T_UP and T_DN

*)
END_FUNCTION_BLOCK
