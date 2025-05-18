FUNCTION_BLOCK SWITCH_X
VAR_INPUT
    IN1 : BOOL;
    IN2 : BOOL;
    IN3 : BOOL;
    IN4 : BOOL;
    IN5 : BOOL;
    IN6 : BOOL;
    t_debounce : TIME := TIME#50ms;
END_VAR

VAR_OUTPUT
    Q1 : BOOL;
    Q2 : BOOL;
    Q3 : BOOL;
    Q4 : BOOL;
    Q5 : BOOL;
    Q6 : BOOL;
    Q31 : BOOL;
    Q41 : BOOL;
    Q51 : BOOL;
    Q61 : BOOL;
    Q32 : BOOL;
    Q42 : BOOL;
    Q52 : BOOL;
    Q62 : BOOL;
END_VAR

VAR
    init : BOOL;
    T1 : OSCAT_BASIC.TOF;
    T2 : OSCAT_BASIC.TOF;
    T3 : OSCAT_BASIC.TOF;
    T4 : OSCAT_BASIC.TOF;
    T5 : OSCAT_BASIC.TOF;
    T6 : OSCAT_BASIC.TOF;
    tx : TIME;
    x1 : BOOL;
    x2 : BOOL;
    E1 : BOOL;
    E2 : BOOL;
END_VAR
(* initialize on startup *)
IF NOT init THEN
	init := TRUE;
	IF t_debounce < t#50ms THEN tx := t#50ms; ELSE tx := t_debounce; END_IF;
	T1(PT := Tx);
	T2(PT := Tx);
	T3(PT := Tx);
	T4(PT := Tx);
	T5(PT := Tx);
	T6(PT := Tx);
ELSE
	Q1 := FALSE;
	Q2 := FALSE;
	Q3 := FALSE;
	Q4 := FALSE;
	Q5 := FALSE;
	Q6 := FALSE;
	Q31 := FALSE;
	Q41 := FALSE;
	Q51 := FALSE;
	Q61 := FALSE;
	Q32 := FALSE;
	Q42 := FALSE;
	Q52 := FALSE;
	Q62 := FALSE;
END_IF;

(* read inputs and debounce *)
	T1(IN := IN1);
	T2(IN := IN2);
	T3(IN := IN3);
	T4(IN := IN4);
	T5(IN := IN5);
	T6(IN := IN6);

(* detect edge of IN1 and IN2 *)
IF t1.Q AND NOT E1 THEN X1 := TRUE; END_IF;
IF t2.Q AND NOT E2 THEN X2 := TRUE; END_IF;

IF t1.Q THEN
	IF t3.q THEN
		q31 := TRUE;
		X1 := FALSE;
	ELSIF t4.q THEN
		q41 := TRUE;
		X1 := FALSE;
	ELSIF t5.q THEN
		q51 := TRUE;
		X1 := FALSE;
	ELSIF t6.q THEN
		q61 := TRUE;
		X1 := FALSE;
	END_IF;
ELSIF t2.Q THEN
	IF t3.q THEN
		q32 := TRUE;
		X2 := FALSE;
	ELSIF t4.q THEN
		q42 := TRUE;
		X2 := FALSE;
	ELSIF t5.q THEN
		q52 := TRUE;
		X2 := FALSE;
	ELSIF t6.q THEN
		q62 := TRUE;
		X2 := FALSE;
	END_IF;
(* in1 was active alone *)
ELSIF NOT T1.Q AND E1 AND X1 THEN
	Q1 := TRUE;
	X1 := FALSE;
ELSIF NOT T2.Q AND E2 AND X2 THEN
	Q2 := TRUE;
	X2 := FALSE;
ELSIF T3.Q THEN
	Q3 := TRUE;
ELSIF T4.Q THEN
	Q4 := TRUE;
ELSIF T5.Q THEN
	Q5 := TRUE;
ELSIF T6.Q THEN
	Q6 := TRUE;
END_IF;

(* save state of in1 and in2 *)
E1 := T1.Q;
E2 := T2.Q;
END_FUNCTION_BLOCK
