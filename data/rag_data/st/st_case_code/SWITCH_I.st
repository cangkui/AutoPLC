FUNCTION_BLOCK SWITCH_I
VAR_INPUT
    set : BOOL;
    in : BOOL;
    rst : BOOL;
    T_debounce : TIME := TIME#10ms;
    T_reconfig : TIME := TIME#1s0ms;
    T_on_max : TIME := TIME#0ms;
END_VAR

VAR_OUTPUT
    Q : BOOL;
END_VAR

VAR
    state : BYTE;
    edge : BOOL;
    r_edge : BOOL;
    T_on : TIME;
    tx : TIME;
END_VAR
(* read system time *)
tx := DWORD_TO_TIME(OSCAT_BASIC.T_PLC_MS());

(* asynchronous set and reset first *)
IF set AND NOT rst THEN
	Q := TRUE;
	T_on := tx;
ELSIF rst THEN
	Q := FALSE;
(* check FOR input edge AND start debounce timer *)
ELSIF (in XOR edge) AND NOT (state = 1) THEN
	state := 1;
	T_on := Tx;
(* edge was detected and debounce time elapsed *)
ELSIF (state = 1) AND (Tx - t_debounce >= T_on) THEN
	state := 2;
	(* react on rising or falling edge depending on r_edge *)
	IF r_edge XOR in THEN Q := NOT Q; END_IF;
(* wait for T_reconfig and adjust r_edge *)
ELSIF state = 2 AND Tx - t_reconfig >= T_on THEN
	r_edge := in;
END_IF;
IF Q AND T_on_max > t#0ms AND Tx >= T_on + T_on_max THEN
	Q := FALSE;
END_IF;
edge := in;

(* revision history

hm	4 aug 2006	rev 1.0
	original version

hm	15.9.2007	rev 1.1
	replaced Time() with T_PLC_MS for compatibility and performance reasons

*)
END_FUNCTION_BLOCK
