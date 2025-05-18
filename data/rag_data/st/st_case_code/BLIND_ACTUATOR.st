FUNCTION_BLOCK BLIND_ACTUATOR
VAR_INPUT
    UP       : BOOL;
    DN       : BOOL;
    S_IN     : BYTE;
    T_UD     : TIME := TIME#10s0ms;
    T_ANGLE  : TIME := TIME#3s0ms;
    T_LOCKOUT : TIME := TIME#100ms;
END_VAR

VAR_OUTPUT
    POS      : BYTE;
    ANG      : BYTE;
    QU       : BOOL;
    QD       : BOOL;
    STATUS    : BYTE;
END_VAR

VAR
    position  : OSCAT_BASIC.RMP_B; 
    angle     : OSCAT_BASIC.RMP_B; 
    lock      : OSCAT_BASIC.INTERLOCK; 
END_VAR
(* make sure only one motor is active at a given time *)
lock(i1 := UP, I2 := DN, TL := T_lockout);

(* ramp up or down to simulate the angle position of the blind slats *)
angle(e := lock.Q1 OR lock.Q2, UP := lock.Q1, PT := T_Angle);
position(e := lock.Q1 AND angle.high OR lock.Q2 AND angle.low, up := lock.Q1, PT := T_UD);

(* set the outputs *)
pos := position.Out;
ang := angle.Out;

(* set the outputs *)
QU := lock.Q1;
QD := lock.Q2;

(* set the status output *)
IF UP AND DN THEN
	 status := 1;			(* error up and down together are not allowed *)
ELSIF UP THEN
	status := 121;
ELSIF DN THEN
	status := 122;
ELSE
	status := S_IN;
END_IF;


(* revision history
hm	29. sep 2007	rev 1.0
	original release

hm	6. oct 2007	rev 1.1
	rearanged outputs

hm	18. oct. 2008	rev 1.2
	changed calls for rmp_B because of change in RMP_B

hm	27. jul. 2009	rev 1.3
	changes status to be 12X

ae	08. mar. 2010 rev 1.4	
	at aPosition

*)
END_FUNCTION_BLOCK
