FUNCTION BCDC_TO_INT : INT
VAR_INPUT
    IN : BYTE;
END_VAR
BCDC_TO_INT := (in AND 16#0F) + (SHR(in,4) * 10);

(* revision history
hm	13.12.2007	rev 1.0
	original version

hm	30.6.2008	rev 1.1
	changed name BCD_TO_INT to BCDC_TO_INT to avoid collision with util.lib

*)
END_FUNCTION
