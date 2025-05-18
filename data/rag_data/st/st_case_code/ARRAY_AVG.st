FUNCTION ARRAY_AVG : REAL
VAR_INPUT
    pt : POINTER TO ARRAY [0..32000] OF REAL;
    size : UINT;
END_VAR

VAR
    i : UINT;
    stop : UINT;
END_VAR
stop := SHR(size,2)-1;
ARRAY_AVG := pt^[0];
FOR i := 1 TO stop DO
	ARRAY_AVG := ARRAY_AVG + pt^[i];
END_FOR;
ARRAY_AVG := ARRAY_AVG / UINT_TO_REAL(stop + 1);


(* revision history
hm	2. oct 2007		rev 1.0
	original version

hm	12. dec 2007	rev 1.1
	chaged code for better performance

hm	16. mar. 2008	rev 1.2
	changed input size to uint

hm	10. mar. 2009	rev 1.3
	added type conversion for compatibility reasons
*)
END_FUNCTION
