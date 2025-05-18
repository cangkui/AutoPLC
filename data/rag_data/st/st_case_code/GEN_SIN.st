FUNCTION_BLOCK GEN_SIN
VAR_INPUT
    PT : TIME;
    AM : REAL := 1.0;
    OS : REAL;
    DL : REAL;
END_VAR

VAR_OUTPUT
    Q : BOOL;
    Out : REAL;
END_VAR

VAR
    tx : TIME;
    last : TIME;
    init : BOOL;
    temp : REAL;
END_VAR
(* read system time and prepare input data *)
tx := DWORD_TO_TIME(T_PLC_MS()) - last;
DL := MODR(dl,1.0);
IF dl < 0.0 THEN dl := 1.0 - dl; END_IF;

(* init section *)
IF NOT init THEN
	init := TRUE;
	last := tx;
	tx := t#0s;
END_IF;

(* add last if one cycle is finished *)
IF tx >= pt THEN
	last := last + pt;
	tx := tx - pt;
END_IF;

(* generate sine wave *)
IF pt > t#0s THEN temp := SIN(math.PI2 * DWORD_TO_REAL(TIME_TO_DWORD(tx + MULTIME(pt, dl))) / DWORD_TO_REAL(TIME_TO_DWORD(pt))); END_IF;
out := am * 0.5 *  temp + os;

(* boolean output Q *)
q := NOT SIGN_R(temp);

(* revision history
hm	22. jan 2007	rev 1.0
	original version

hm	17 sep 2007	rev 1.1
	replaced time() with t_plc_ms for compatibilitx reasons

hm	27. nov 2007	rev 1.2
	avoid divide by 0 when pt = 0

hm	6. jan 2008		rev 1.3
	improved performance

hm	16. mar. 2008	rev 1.4
	added type conversion to avoid warnings under codesys 3.0

hm	18. oct. 2008	rev 1.5
	using math constants

hm	11. mar. 2009	rev 1.6
	changed real constants to use dot syntax

*)
END_FUNCTION_BLOCK
