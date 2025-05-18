FUNCTION TANK_VOL1 : REAL
VAR_INPUT
    TR : REAL;
    TL : REAL;
    H : REAL;
END_VAR
TANK_VOL1 := OSCAT_BASIC.CIRCLE_SEG(TR, H) * TL;


(* revision histroy
hm	10. mar 2010	rev 1.0
	original version

*)
END_FUNCTION
