FUNCTION_BLOCK HEAT_METER
VAR_INPUT
    TF : REAL;
    TR : REAL;
    LPH : REAL;
    E : BOOL;
    RST : BOOL;
    CP : REAL; // input_constant
    Density : REAL; // input_constant
    Content : REAL; // input_constant
    Pulse_Mode : BOOL; // input_constant
    Return_meter : BOOL; // input_constant
    AVG_time : TIME := TIME#5s0ms; // input_constant
END_VAR

VAR_OUTPUT
    C : REAL;
END_VAR

VAR_IN_OUT
    Y : REAL;
END_VAR

VAR
    tx : DWORD;
    last : DWORD;
    int1 : OSCAT_BASIC.FT_INT2;
    edge : BOOL;
    x : REAL;
    init : BOOL;
    y_last : REAL;
END_VAR
IF rst THEN
	int1(rst := TRUE);
	int1.rst := FALSE;
	C := 0.0;
	Y := 0.0;
ELSIF e THEN
	X := (WATER_DENSITY(SEL(return_meter, TF, TR), FALSE) * (WATER_ENTHALPY(TF) - WATER_ENTHALPY(TR)) * (1.0 - content) + CP * Density * content * (TF-TR));
END_IF;

(* integrate or add consumption *)
int1(run := NOT pulse_mode AND e, in := X * LPH * 2.77777777777E-4);
IF pulse_mode THEN
	IF NOT edge AND E THEN Y := Y + X * LPH; END_IF;
ELSE
	Y := int1.Out;
END_IF;

(* store the value of e *)
edge := e;

(* read system_time *)
tx := OSCAT_BASIC.T_PLC_MS();

(* only init at startup necessary *)
IF NOT init THEN
	init := TRUE;
	last := tx;
END_IF;

(* calculate the current consumption *)
IF (tx - last >= TIME_TO_DWORD(AVG_TIME)) AND (avg_time > T#0s) THEN
	last := tx;
	C := (Y - Y_last) * 3.6E6 / DWORD_TO_REAL(TIME_TO_DWORD(AVG_TIME));
	Y_last := Y;
END_IF;



(* revision history
hm	16. nov. 2007	rev 1.0
	original version

hm	24. dec. 2007	rev 1.1
	added config variable return_meter
	added output C for current consumption

hm	6. mar. 2008	rev 1.2
	corrected an error in pulse mode

hm	16. mar. 2008	rev 1.3
	added type conversion to avoid warnings under codesys 3.0

hm	21. oct. 2008	rev 1.4
	improved code

hm	6. nov. 2008	rev 1.5
	changed to FT_INT2 to avoid overrun at hig values

hm	8. feb. 2009	rev 1.6
	changes Y to be I/O

hm	23. mar. 2009	rev 1.7
	real constants updated to new systax using dot
	corected error in formula

hm	27. jul 2009	rev 1.8
	output C is now calculated in J/h

hm	13. nov. 2009	rev 1.9
	output C is now calculated correctly

hm	23. jan. 2011	rev 1.10
	corrected an error with pulse_mode
*)
END_FUNCTION_BLOCK
