FUNCTION_BLOCK MOON_PHASE
VAR_INPUT
    XDT : DT;
    SCALE : BYTE := 12;
    UPDATE : TIME := TIME#60m0s;
END_VAR

VAR_OUTPUT
    PHASE : BYTE;
END_VAR

VAR
    x : UDINT;
    last_dt : DT;
END_VAR
IF XDT - last_dt > UPDATE THEN
	x := (DT_TO_UDINT(XDT) - 603240) MOD 2551392;
	PHASE := UDINT_TO_BYTE(SCALE * x / 2551392) ;
	last_dt := XDT;
END_IF;

(* revision history
ks	15.may. 2011	rev 1.0
	original version

*)
END_FUNCTION_BLOCK
