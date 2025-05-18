FUNCTION TANK_VOL2 : REAL
VAR_INPUT
    TR : REAL;
    H : REAL;
END_VAR
TANK_VOL2 := OSCAT_BASIC.math.PI * H * H * (TR - H/3.0);

(* revision histroy
hm	10. mar 2010	rev 1.0
	original version

*)
END_FUNCTION
