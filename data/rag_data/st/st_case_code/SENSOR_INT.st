FUNCTION SENSOR_INT : REAL
VAR_INPUT
    Voltage : REAL;
    Current : REAL;
    RP : REAL;
    RS : REAL;
END_VAR

VAR
    RG : REAL;
END_VAR
IF current <> 0.0 THEN
	RG := voltage / current;
	SENSOR_INT := RP * ( RG - RS) / ( RP + RS - RG);
END_IF;

(* revisaion history

hm 20.1.2007		rev 1.1
	deleted input R0 which was not needed.

hm	2. dec 2007	rev 1.2
	corrected an error in formula and improoved speed

hm	11. mar. 2009	rev 1.3
	changed real constants to use dot syntax

*)
END_FUNCTION
