FUNCTION_BLOCK DRIVER_1
VAR_INPUT
    Toggle_Mode : BOOL;  
    Timeout : TIME;  
    SET : BOOL;  
    IN : BOOL;  
    RST : BOOL;  
END_VAR

VAR_OUTPUT
    Q : BOOL;  
END_VAR

VAR
    off : TON;  
    edge : BOOL;  
END_VAR
IF off.Q THEN Q := FALSE; END_IF;
IF rst THEN
	Q := FALSE;
ELSIF set THEN
	Q := TRUE;
ELSIF IN AND NOT edge THEN
	IF toggle_mode THEN q := NOT Q; ELSE q := TRUE; END_IF;
END_IF;
edge := in;
IF timeout > t#0s THEN off(in := Q, PT := Timeout); END_IF;


(* revision history
hm	2. jan 2008		rev 1.0
	original version

*)
END_FUNCTION_BLOCK
