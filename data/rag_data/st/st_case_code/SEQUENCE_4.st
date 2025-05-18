FUNCTION_BLOCK SEQUENCE_4
VAR_INPUT
    in0 : BOOL := TRUE;
    in1 : BOOL := TRUE;
    in2 : BOOL := TRUE;
    in3 : BOOL := TRUE;
    start : BOOL;
    rst : BOOL;
    wait0 : TIME;
    delay0 : TIME;
    wait1 : TIME;
    delay1 : TIME;
    wait2 : TIME;
    delay2 : TIME;
    wait3 : TIME;
    delay3 : TIME;
    stop_on_error : BOOL;
END_VAR

VAR_OUTPUT
    Q0 : BOOL;
    Q1 : BOOL;
    Q2 : BOOL;
    Q3 : BOOL;
    QX : BOOL;
    run : BOOL;
    step : INT := -1;
    status : BYTE;
END_VAR

VAR
    last : TIME;
    edge : BOOL;
    tx : TIME;
    init : BOOL;
END_VAR
(* read sps timer *)
tx := DWORD_TO_TIME(T_PLC_MS());

(* initialize on startup *)
IF NOT init THEN
	last := tx;
	init := TRUE;
	status := 110;
END_IF;

(* asynchronous reset *)
IF rst THEN
	step := -1;
	Q0 := 0;
	Q1 := 0;
	Q2 := 0;
	Q3 := 0;
	status := 110;
	run := 0;

(* edge on start input restarts the sequencer *)
ELSIF start AND NOT edge THEN
	step := 0;
	last := tx;
	status := 111;
	Q0 := 0;
	Q1 := 0;
	Q2 := 0;
	Q3 := 0;
	run := 1;
END_IF;
edge := start;

(* check if stop on status is necessary *)
IF status > 0 AND status < 100 AND stop_on_error THEN RETURN; END_IF;

(* sequence is running *)
IF run AND step = 0 THEN
	IF NOT q0 AND in0 AND tx - last <= wait0 THEN
		Q0 := TRUE;
		last := tx;
	ELSIF NOT q0 AND tx - last > wait0 THEN
		status := 1;
		run := FALSE;
	ELSIF q0 AND tx - last >= delay0 THEN
		step := 1;
		last := tx;
	END_IF;
END_IF;
IF run AND step = 1 THEN
	IF NOT q1 AND in1 AND tx - last <= wait1 THEN
		Q0 := FALSE;
		Q1 := TRUE;
		last := tx;
	ELSIF NOT q1 AND Tx - last > wait1 THEN
		status := 2;
		q0 := FALSE;
		run := FALSE;
	ELSIF q1 AND tx - last >= delay1 THEN
		step := 2;
		last := tx;
	END_IF;
END_IF;
IF run AND step = 2 THEN
	IF NOT q2 AND in2 AND tx - last <= wait2 THEN
		Q1 := FALSE;
		Q2 := TRUE;
		last := tx;
	ELSIF NOT q2 AND Tx - last > wait2 THEN
		status := 3;
		q1 := FALSE;
		run := FALSE;
	ELSIF q2 AND tx - last >= delay2 THEN
		step := 3;
		last := tx;
	END_IF;
END_IF;
IF run AND step = 3 THEN
	IF NOT q3 AND in3 AND tx - last <= wait3 THEN
		Q2 := FALSE;
		Q3 := TRUE;
		last := tx;
	ELSIF NOT q3 AND Tx - last > wait3 THEN
		status := 4;
		q2 := FALSE;
		run := FALSE;
	ELSIF q3 AND tx - last >= delay3 THEN
		step := -1;
		q3 := FALSE;
		run := FALSE;
		status := 110;
	END_IF;
END_IF;
QX := q0 OR q1 OR q2 OR q3;

(*
hm 1.10.06		rev 1.1
	corrected delay logic to be after event and not before
	added any output

hm 1.12.06		rev 1.2
	corrected failure in sequence logic.
	added init at startup to prevent from initial error after start.

hm 17.1.2007		rev 1.3
	changed output fault to status for better compatibility with other modules (ESR).
	added stop on error functionality and setup variable
	default for inputs in0..3 is true.
	renamed variable state to step

hm	17.sep 2007	rev 1.4
	replaced time() with T_PLC_MS() for compatibility reasons

hm	13. mar. 2009	rev 1.5
	renamed output any to qx for compatibility resons

*)
END_FUNCTION_BLOCK
