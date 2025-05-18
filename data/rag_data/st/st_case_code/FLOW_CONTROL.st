FUNCTION_BLOCK FLOW_CONTROL
VAR_INPUT
    IN : BOOL;
    REQ : BOOL;
    ENQ : BOOL;
    RST : BOOL;
    T_AUTO : TIME := TIME#60m0s0ms;
    T_DELAY : TIME := TIME#1380m0s0ms;
END_VAR

VAR_OUTPUT
    Q : BOOL;
    STATUS : BYTE;
END_VAR

VAR
    timer : TP_1D;
END_VAR
STATUS := 100;
IF RST THEN
	Q := FALSE;
	timer(rst := TRUE);
	timer.RST := FALSE;
	STATUS := 103;
ELSIF ENQ THEN
	IF IN THEN
		status := 101;
	END_IF;
	IF REQ THEN
		(* timer will generate a timed pulse after TP goes high *)
		timer.PT1 := T_AUTO;
		timer.PTD := T_DELAY;
		timer.IN := TRUE;
		STATUS := 102;
	END_IF;
END_IF;


(* set output and run timer *)
timer();
timer.IN := FALSE;
Q := (IN AND ENQ) OR timer.Q;



(* revision history
hm 	28. jun. 2008 	rev 1.0
	original version

*)
END_FUNCTION_BLOCK
