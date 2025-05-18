FUNCTION MANUAL : BOOL
VAR_INPUT
    IN : BOOL;
    ON : BOOL;
    OFF : BOOL;
END_VAR
MANUAL := NOT OFF AND (IN OR ON);



(* revision history
hm	21. nov 2008	rev 1.0
	original version


*)
END_FUNCTION
