FUNCTION_BLOCK MESSAGE_4R
VAR_INPUT
    M0 : STRING(STRING_LENGTH);
    M1 : STRING(STRING_LENGTH);
    M2 : STRING(STRING_LENGTH);
    M3 : STRING(STRING_LENGTH);
    MM : INT := 3;
    ENQ : BOOL := TRUE;
    CLK : BOOL := TRUE;
    T1 : TIME := T#3s0ms;
END_VAR

VAR_OUTPUT
    MX : STRING(STRING_LENGTH);
    MN : INT;
    TR : BOOL;
END_VAR

VAR
    timer : TON;
    edge : BOOL;
END_VAR
(* check for rising edge on EN *)
TR := FALSE;
IF ENQ THEN
	IF (NOT edge AND clk) OR timer.q THEN
			MN := INC1(MN, MM);
			TR := TRUE;
			timer(in := FALSE);
			CASE MN OF
				0 : MX := M0;
				1 : MX := M1;
				2 : MX := M2;
				3 : MX := M3;
			END_CASE;
	END_IF;
	edge := clk;
	timer( in := CLK, pt := T1);
ELSE
	MX := '';
	MN := 0;
END_IF;




(* revision history
hm	8. oct. 2008	rev 1.0
	original version

hm	27. oct. 2008	rev 1.1
	changed _INC to INC1

*)
END_FUNCTION_BLOCK
