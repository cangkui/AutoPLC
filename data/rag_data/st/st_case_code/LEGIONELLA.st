FUNCTION_BLOCK LEGIONELLA
VAR_INPUT
    manual: BOOL;
    temp_boiler: REAL;
    temp_return: REAL := 100.0;
    DT_in: DT;
    rst: BOOL;
    T_start: TOD := TIME_OF_DAY#3:0;
    day: INT := 7;
    temp_set: REAL := 70.0;
    temp_offset: REAL := 10.0;
    temp_hys: REAL := 5.0;
    t_max_heat: TIME := TIME#10m0s0ms;
    t_max_ret: TIME := TIME#10m0s0ms;
    tp_0: TIME := TIME#5m0s0ms;
    TP_1: TIME := TIME#5m0s0ms;
    tp_2: TIME := TIME#5m0s0ms;
    tp_3: TIME := TIME#5m0s0ms;
    tp_4: TIME := TIME#5m0s0ms;
    tp_5: TIME := TIME#5m0s0ms;
    tp_6: TIME := TIME#5m0s0ms;
    tp_7: TIME := TIME#5m0s0ms;
END_VAR

VAR_OUTPUT
    Heat: BOOL;
    pump: BOOL;
    valve0: BOOL;
    valve1: BOOL;
    valve2: BOOL;
    valve3: BOOL;
    valve4: BOOL;
    valve5: BOOL;
    valve6: BOOL;
    valve7: BOOL;
    run: BOOL;
    Status: BYTE;
END_VAR

VAR
    X1: TIMER_1;
    x2: OSCAT_BASIC.SEQUENCE_8;
    x3: OSCAT_BASIC.HYST_1;
    init: BOOL;
END_VAR
(* startup initialization *)
IF NOT init THEN
	init := TRUE;
	X1.day := SHR(BYTE#128,day);
	X1.start := T_start;
	X3.low := Temp_offset + temp_set;
	X3.high := Temp_hys + X3.low;
	X2.wait0 := T_max_heat;
	X2.delay0 := TP_0;
	X2.delay1 := TP_1;
	X2.delay2 := TP_2;
	X2.delay3 := TP_3;
	X2.delay4 := TP_4;
	X2.delay5 := TP_5;
	X2.delay6 := TP_6;
	X2.delay7 := TP_7;
	X2.wait1 := T_max_ret;
	X2.wait2 := T_max_ret;
	X2.wait3 := T_max_ret;
	X2.wait4 := T_max_ret;
	X2.wait5 := T_max_ret;
	X2.wait6 := T_max_ret;
	X2.wait7 := T_max_ret;
	X2();
END_IF;

(* oerational code *)
X1(DTi := DT_in);
IF X1.Q OR MANUAL OR x2.run THEN
	X3(in := temp_boiler);
	X2.in0 := X3.Q OR x3.win;
	X2.in1 := temp_return >= temp_set;
	X2.in2 := x2.in1;
	X2.in3 := x2.in1;
	X2.in4 := x2.in1;
	X2.in5 := x2.in1;
	X2.in6 := x2.in1;
	X2.in7 := x2.in1;
	X2.rst := rst;
	X2(start := X1.Q OR MANUAL);
	run := x2.run;
	pump := x2.qx;
	Heat := NOT X3.Q AND x2.run;
	valve0 := X2.Q0;
	valve1 := X2.Q1;
	valve2 := X2.Q2;
	valve3 := X2.Q3;
	valve4 := X2.Q4;
	valve5 := X2.Q5;
	valve6 := X2.Q6;
	valve7 := X2.Q7;
	pump := X2.qx;
	status := X2.status;
ELSE;
	X2(start :=0);
	status := x2.status;
END_IF;

(* revision history

hm 17.jan 2007	rev 1.1
	rewritten to st for higher performance
	changes eeror output to esr compatible status output

hm	23. 10.2008	rev 1.2
	update code to use new version of timer_1

hm	13. mar. 2009	rev 1.3
	real constants updated to new systax using dot
	using new module sequence_8
*)
END_FUNCTION_BLOCK
