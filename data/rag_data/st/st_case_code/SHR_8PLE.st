FUNCTION_BLOCK SHR_8PLE
VAR_INPUT
    Din : BOOL;
    Dload : BYTE;
    CLK : BOOL;
    UP : BOOL := TRUE; // Initial Value = 1
    load : BOOL;
    RST : BOOL;
END_VAR

VAR_OUTPUT
    DOut : BOOL;
END_VAR

VAR
    edge : BOOL := TRUE; // Initial Value = 1
    register : BYTE;
END_VAR
(* flankenerkennung clk wird high und edge war high reset ist nicht aktiv und set ist nicht aktiv *)
IF CLK AND edge AND NOT rst THEN
	edge := FALSE;	(* flanke wurde erkannt und weitere flankenerkennung wird verhindert bis edge wieder true ist *)
	(* hier ist der code für das flankenevent *)
	IF UP THEN						(*shift up *)
		register := SHL(register,1);
		register.0 := Din;
		Dout := register.7;
	ELSE								(* shift down *);
		register := SHR(register,1);
		register.7 := Din;
		Dout := register.0;
	END_IF;
	IF load THEN							(* the byte on Din will be loaded if load = true *)
		register := Dload;
		IF up THEN Dout := register.7; ELSE Dout := register.0; END_IF;
	END_IF;
END_IF;
IF NOT clk THEN edge := TRUE; END_IF;	(* sobald clk wieder low wird warten auf nächste flanke *)
IF rst THEN									(* wenn reset aktiv dann ausgang rücksetzen *)
	register := 0;
	Dout := FALSE;
END_IF;
END_FUNCTION_BLOCK
