FUNCTION_BLOCK ESR_MON_B8
VAR_INPUT
    s0 : BOOL;
    s1 : BOOL;
    s2 : BOOL;
    s3 : BOOL;
    s4 : BOOL;
    s5 : BOOL;
    s6 : BOOL;
    s7 : BOOL;
    DT_in : DT;
    a0 : STRING[10];
    a1 : STRING[10];
    a2 : STRING[10];
    a3 : STRING[10];
    a4 : STRING[10];
    a5 : STRING[10];
    a6 : STRING[10];
    a7 : STRING[10];
END_VAR

VAR_OUTPUT
    ESR_Flag : BOOL;
END_VAR

VAR_IN_OUT
    ESR_Out : ARRAY[0..3] OF esr_data;
END_VAR

VAR
    x0 : BOOL;
    x1 : BOOL;
    x2 : BOOL;
    x3 : BOOL;
    x4 : BOOL;
    x5 : BOOL;
    x6 : BOOL;
    x7 : BOOL;
    tx : TIME;
    cnt : INT;
END_VAR
(* read system timer *)
tx := DWORD_TO_TIME(T_PLC_MS());

ESR_Flag := FALSE;
esr_out[3].typ := 0;
esr_out[2].typ := 0;
esr_out[1].typ := 0;
esr_out[0].typ := 0;
cnt := 0;


(* check if inputs have chaged and fill buffer *)
IF s0 <> X0 THEN
	esr_out[cnt].typ := 10 + BOOL_TO_BYTE(s0);
	esr_out[cnt].adress := a0;
	esr_out[cnt].DS := DT_in;
	esr_out[cnt].TS := TX;
	X0 := S0;
	cnt := cnt + 1;
	esr_flag := TRUE;
END_IF;
IF s1 <> X1 THEN
	esr_out[cnt].typ := 10 + BOOL_TO_BYTE(s1);
	esr_out[cnt].adress := a1;
	esr_out[cnt].DS := DT_in;
	esr_out[cnt].TS := TX;
	X1 := S1;
	cnt := cnt + 1;
	esr_flag := TRUE;
END_IF;
IF s2 <> X2 THEN
	esr_out[cnt].typ := 10 + BOOL_TO_BYTE(s2);
	esr_out[cnt].adress := a2;
	esr_out[cnt].DS := DT_in;
	esr_out[cnt].TS := TX;
	X2 := S2;
	cnt := cnt + 1;
	esr_flag := TRUE;
END_IF;
IF s3 <> X3 THEN
	esr_out[cnt].typ := 10 + BOOL_TO_BYTE(s3);
	esr_out[cnt].adress := a3;
	esr_out[cnt].DS := DT_in;
	esr_out[cnt].TS := TX;
	X3 := S3;
	cnt := cnt + 1;
	esr_flag := TRUE;
END_IF;
IF s4 <> X4 AND cnt < 4 THEN
	esr_out[cnt].typ := 10 + BOOL_TO_BYTE(s4);
	esr_out[cnt].adress := a4;
	esr_out[cnt].DS := DT_in;
	esr_out[cnt].TS := TX;
	X4 := S4;
	cnt := cnt + 1;
	esr_flag := TRUE;
END_IF;
IF s5 <> X5  AND cnt < 4 THEN
	esr_out[cnt].typ := 10 + BOOL_TO_BYTE(s5);
	esr_out[cnt].adress := a5;
	esr_out[cnt].DS := DT_in;
	esr_out[cnt].TS := TX;
	X5 := S5;
	cnt := cnt + 1;
	esr_flag := TRUE;
END_IF;
IF s6 <> X6  AND cnt < 4 THEN
	esr_out[cnt].typ := 10 + BOOL_TO_BYTE(s6);
	esr_out[cnt].adress := a6;
	esr_out[cnt].DS := DT_in;
	esr_out[cnt].TS := TX;
	X6 := S6;
	cnt := cnt + 1;
	esr_flag := TRUE;
END_IF;
IF s7 <> X7  AND cnt < 4 THEN
	esr_out[cnt].typ := 10 + BOOL_TO_BYTE(s7);
	esr_out[cnt].adress := a7;
	esr_out[cnt].DS := DT_in;
	esr_out[cnt].TS := TX;
	X7 := S7;
	cnt := cnt + 1;
	esr_flag := TRUE;
END_IF;

(* revision history
hm	26. jan 2007	rev 1.0
	original version

hm	17. sep 2007	rev 1.1
	replaced time() with T_PLC_MS() for compatibility reasons

hm	22. oct. 2008	rev 1.2
	optimized code

hm	1.dec. 2009	rev 1.3
	changed esr_out to be I/O

*)
END_FUNCTION_BLOCK
