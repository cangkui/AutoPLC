FUNCTION IS_SORTED : BOOL
VAR_INPUT
    pt : ^ ARRAY [0..32000] OF REAL; // Pointer to an array of REAL
    size : UINT; // Size variable of type UINT
END_VAR

VAR
    stop : INT; // Local variable of type INT
    i : INT; // Local variable of type INT
END_VAR
stop := UINT_TO_INT(SHR(size,2)-2);
FOR i := 0 TO stop DO
	IF pt^[i] > pt^[i+1] THEN
		IS_SORTED := FALSE;
		RETURN;
	END_IF;
END_FOR;

IS_SORTED := TRUE;

(* revision history

hm 	29. mar. 2008 	rev 1.0
	original version

hm	4. apr. 2008	rev 1.1
	added type conversion to avoid warnings under codesys 3.0
*)
END_FUNCTION
