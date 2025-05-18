FUNCTION_BLOCK CALIBRATE
VAR_INPUT
    X : REAL;
    CO : BOOL;
    CS : BOOL;
    Y_Offset : REAL; (* input_constant *)
    Y_Scale : REAL;  (* input_constant *)
END_VAR

VAR_OUTPUT
    Y : REAL;
END_VAR

VAR RETAIN
    offset : REAL;
    Scale : REAL := 1.0; (* initial value *)
END_VAR
(* check for calibration *)
IF CO THEN
	OFFSET := Y_Offset - X;
ELSIF CS THEN
	SCALE := Y_scale / (X + OFFSET);
END_IF;
(* calculate output *)
Y := (X + OFFSET) * SCALE;


(* revision history
hm 22.2.2007		rev 1.2
	changed VAR RETAIN PERSISTENT to VAR RETAIN for better compatibility
	wago lon contollers do not support persisitent

hm	11. mar. 2009	rev 1.3
	changed real constants to use dot syntax

*)
END_FUNCTION_BLOCK
