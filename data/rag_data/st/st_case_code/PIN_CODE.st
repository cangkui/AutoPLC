FUNCTION_BLOCK PIN_CODE
VAR_INPUT
    CB : BYTE;
    E : BOOL;
    PIN : STRING[8]; // input_constant attribute is ignored in ST declaration
END_VAR

VAR_OUTPUT
    TP : BOOL;
END_VAR

VAR
    POS : INT := 1; // Initialized to 1
END_VAR
tp := FALSE;
IF e THEN
	IF CB = CODE(pin, pos) THEN
		pos := pos + 1;
		IF pos > LEN(PIN) THEN
			(* proper code is detected *)
			TP := TRUE;
			pos := 1;
		END_IF;
	ELSE
		pos := 1;
	END_IF;
END_IF;


(* revision history
hm	30. oct. 2008	rev 1.0		
	original version 


*)
END_FUNCTION_BLOCK
