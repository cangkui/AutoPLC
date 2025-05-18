FUNCTION_BLOCK DRIVER_4
VAR_INPUT
    Toggle_Mode : BOOL;
    Timeout : TIME;
    SET : BOOL;
    IN0 : BOOL;
    IN1 : BOOL;
    IN2 : BOOL;
    IN3 : BOOL;
    RST : BOOL;
END_VAR

VAR_OUTPUT
    Q0 : BOOL;
    Q1 : BOOL;
    Q2 : BOOL;
    Q3 : BOOL;
END_VAR

VAR
    d0 : DRIVER_1;
    d1 : DRIVER_1;
    d2 : DRIVER_1;
    d3 : DRIVER_1;
END_VAR
D0(Set:=set, in:=in0, rst:=rst, toggle_mode:=toggle_mode, timeout:=timeout);
D1(Set:=set, in:=in1, rst:=rst, toggle_mode:=toggle_mode, timeout:=timeout);
D2(Set:=set, in:=in2, rst:=rst, toggle_mode:=toggle_mode, timeout:=timeout);
D3(Set:=set, in:=in3, rst:=rst, toggle_mode:=toggle_mode, timeout:=timeout);
Q0 := D0.Q;
Q1 := D1.Q;
Q2 := D2.Q;
Q3 := D3.Q;


(* revision history
hm	2. jan 2008		rev 1.0
	original version

*)
END_FUNCTION_BLOCK
