FUNCTION_BLOCK SH
VAR_INPUT
    in : REAL;    (* Input Real Variable *)
    CLK : BOOL;   (* Clock Input Variable *)
END_VAR

VAR_OUTPUT
    out : REAL;   (* Output Real Variable *)
    trig : BOOL;  (* Output BOOL Variable *)
END_VAR

VAR
    edge : BOOL;  (* Local BOOL Variable *)
END_VAR
IF clk AND NOT edge THEN
	out := in;
	trig := TRUE;
ELSE;
	trig := FALSE;
END_IF;
edge := clk;

(* revision history

hm 16.1.2007	rev 1.1
	added trig output

*)
END_FUNCTION_BLOCK
