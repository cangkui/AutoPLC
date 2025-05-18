FUNCTION_BLOCK FT_PIWL
VAR_INPUT
    IN : REAL;
    KP : REAL := 1.0;
    KI : REAL := 1.0;
    LIM_L : REAL := -1E+38;
    LIM_H : REAL := 1E+38;
    RST : BOOL;
END_VAR

VAR_OUTPUT
    Y : REAL;
    LIM : BOOL;
END_VAR

VAR
    init : BOOL;
    tx : DWORD;
    tc : REAL;
    t_last : DWORD;
    in_last : REAL;
    i : REAL;
    p : REAL;
END_VAR
(* initialize at power_up *)
IF NOT init OR RST THEN
	init := TRUE;
	in_last := in;
	t_last := T_PLC_US();
	i := 0.0;
	tc := 0.0;
ELSE
	(* read last cycle time in Microseconds *)
	tx := T_PLC_US();
	tc := DWORD_TO_REAL(tx - t_last);
	t_last := tx;

	(* calculate proportional part *)
	p := KP * IN;

	(* run integrator *)
	i := (IN + in_last) * 5.0E-7 * KI * tc + i;
	in_last := IN;

	(* calculate output Y *)
	Y := p + i;

	(* check output for limits *)
	IF Y >= LIM_H THEN
		Y := LIM_H;
		IF ki <> 0.0 THEN
			i := LIM_H - p;
		ELSE
			i := 0.0;
		END_IF;
		LIM := TRUE;
	ELSIF Y <= LIM_L THEN
		Y := LIM_L;
		IF ki <> 0.0 THEN
			i := LIM_L - p;
		ELSE
			i := 0.0;
		END_IF;
		LIM := TRUE;
	ELSE
		LIM := FALSE;
	END_IF;
END_IF;




(* revision history
hm 13. jun. 2008 	rev 1.0
	original version

hm	27. oct. 2008	rev 1.1
	integrator will not be adjusted when ki = 0

hm	25. jan 2009	rev 1.2
	module will also work with negative K

hm	11. mar. 2009	rev 1.3
	real constants updated to new systax using dot

*)
END_FUNCTION_BLOCK
