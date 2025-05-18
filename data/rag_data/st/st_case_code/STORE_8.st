FUNCTION_BLOCK STORE_8
VAR_INPUT
    Set : BOOL;
    D0 : BOOL;
    D1 : BOOL;
    D2 : BOOL;
    D3 : BOOL;
    D4 : BOOL;
    D5 : BOOL;
    D6 : BOOL;
    D7 : BOOL;
    Clr : BOOL;
    Rst : BOOL;
END_VAR

VAR_OUTPUT
    Q0 : BOOL;
    Q1 : BOOL;
    Q2 : BOOL;
    Q3 : BOOL;
    Q4 : BOOL;
    Q5 : BOOL;
    Q6 : BOOL;
    Q7 : BOOL;
END_VAR

VAR
    edge : BOOL;
END_VAR
IF rst OR set THEN
	q0 := NOT rst;
	q1 := q0;
	q2 := q0;
	q3 := q0;
	q4 := q0;
	q5 := q0;
	q6 := q0;
	q7 := q0;
ELSE
	IF D0 THEN Q0 := TRUE; END_IF;
	IF D1 THEN Q1 := TRUE; END_IF;
	IF D2 THEN Q2 := TRUE; END_IF;
	IF D3 THEN Q3 := TRUE; END_IF;
	IF D4 THEN Q4 := TRUE; END_IF;
	IF D5 THEN Q5 := TRUE; END_IF;
	IF D6 THEN Q6 := TRUE; END_IF;
	IF D7 THEN Q7 := TRUE; END_IF;

	IF clr AND NOT edge THEN
		IF q0 THEN q0 := FALSE;
		ELSIF q1 THEN q1 := FALSE;
		ELSIF q2 THEN q2 := FALSE;
		ELSIF q3 THEN q3 := FALSE;
		ELSIF q4 THEN q4 := FALSE;
		ELSIF q5 THEN q5 := FALSE;
		ELSIF q6 THEN q6 := FALSE;
		ELSE q7 := FALSE;
		END_IF;
	END_IF;
	edge := clr;
END_IF;


(* revision history
hm	25.12.2007	rev 1.0
	original version

hm	30. oct. 2008	rev 1.1
	optimized performance

hm	14. mar. 2009	rev 1.2
	removed double assignments

*)
END_FUNCTION_BLOCK
