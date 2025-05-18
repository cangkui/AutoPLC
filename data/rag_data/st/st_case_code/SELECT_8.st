FUNCTION_BLOCK SELECT_8
VAR_INPUT
    E: BOOL;
    SET: BOOL;
    IN: BYTE;
    UP: BOOL;
    DN: BOOL;
    RST: BOOL;
END_VAR

VAR_OUTPUT
    Q0: BOOL;
    Q1: BOOL;
    Q2: BOOL;
    Q3: BOOL;
    Q4: BOOL;
    Q5: BOOL;
    Q6: BOOL;
    Q7: BOOL;
    STATE: INT;
END_VAR

VAR
    last_up: BOOL;
    last_dn: BOOL;
END_VAR
IF rst THEN
	state := 0;
ELSIF set THEN
	state := IN;
ELSIF up AND NOT last_up THEN
	state := inc(state,1,7);
ELSIF dn AND NOT last_dn THEN
	state := inc(state,-1,7);
END_IF;
last_UP := UP;
last_DN := DN;

Q0 := FALSE;
Q1 := FALSE;
Q2 := FALSE;
Q3 := FALSE;
Q4 := FALSE;
Q5 := FALSE;
Q6 := FALSE;
Q7 := FALSE;

IF E THEN
	CASE state OF
		0: Q0 := TRUE;
		1: Q1 := TRUE;
		2: Q2 := TRUE;
		3: Q3 := TRUE;
		4: Q4 := TRUE;
		5: Q5 := TRUE;
		6: Q6 := TRUE;
		7: Q7 := TRUE;
	END_CASE;
END_IF;

(* revision history
hm	16. jan 2008	rev 1.0
	original version

hm	27. feb. 2009	rev 1.1
	renamed input en to e
*)
END_FUNCTION_BLOCK
