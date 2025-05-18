FUNCTION_BLOCK ACTUATOR_A
VAR_INPUT
    I1 : BYTE;
    IS : BOOL;
    I2 : BYTE;
    RV : BOOL;
    DX : BOOL;
    RUNTIME : TIME;
    SELF_ACT_TIME : TIME;
    OUT_MIN : WORD;
    OUT_MAX : WORD;
END_VAR

VAR_OUTPUT
    Y : WORD;
END_VAR

VAR
    timer : OSCAT_BASIC.CYCLE_4;
    dx_edge : BOOL;
END_VAR
(* run the system timer, if self_act_time = t#0s then state is set to 0 *)
timer(T0 := RUNTIME, T1 := RUNTIME, T3 := SELF_ACT_TIME, sl := DX AND NOT dx_edge, sx := 0, S0 := SELF_ACT_TIME > t#0s);
dx_edge := dx;

CASE timer.STATE OF
	0: 	(* self act min value *)
		Y := OUT_MIN;
	1:	(* self_act max value *)
		Y := OUT_MAX;
	3:	(* normal operation *)
		IF rv THEN
			Y := OUT_MAX - (OUT_MAX - OUT_MIN) * SEL(IS, I1, I2) / 255;
		ELSE
			Y := (OUT_MAX - OUT_MIN) * SEL(IS, I1, I2) / 255 + OUT_MIN;
		END_IF;
END_CASE;




(*
hm 	8. nov. 2008	rev 1.0
	original version

*)
END_FUNCTION_BLOCK
