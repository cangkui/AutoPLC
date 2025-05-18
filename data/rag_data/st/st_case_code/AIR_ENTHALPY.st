FUNCTION AIR_ENTHALPY : REAL
VAR_INPUT
    T : REAL;
    RH : REAL;
END_VAR

VAR CONSTANT
    CPL : REAL := 1.00482;
    CW : REAL := 1.86;
    LH : REAL := 2500.78;
END_VAR
AIR_ENTHALPY := CPL * T + (CW * T + LH) * DEW_CON(RH,T);

(* revision history

hm	31.10.2007	rev 1.0
	original version

hm	2.11.2007		rev 1.1
	deleted unused variable RF

hm	2.12.2007		rev 1.2
	changed code for better performance
*)
END_FUNCTION
