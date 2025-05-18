FUNCTION BUILDING_VERSION : DWORD
VAR_INPUT
    IN : BOOL;  
END_VAR
IF in THEN
	BUILDING_VERSION := DATE_TO_DWORD(D#2011-02-3);
ELSE
	BUILDING_VERSION := 100;
END_IF;

(* revision history
hm	6. oct 2006	rev 1.0
	original version

hm	16. dec 2007	rev 1.1
	added possibility to return date and version depending on IN.

*)
END_FUNCTION
