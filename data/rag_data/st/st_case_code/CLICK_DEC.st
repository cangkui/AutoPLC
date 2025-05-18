FUNCTION_BLOCK CLICK_DEC
VAR_INPUT
    IN : BOOL;
    TC : TIME;
END_VAR

VAR_OUTPUT
    Q0 : BOOL;
    Q1 : BOOL;
    Q2 : BOOL;
    Q3 : BOOL;
END_VAR

VAR
    tx : TP;      (* Derived type *)
    edge : BOOL;
    cnt : INT := -1; (* Initial value *)
END_VAR
(* Q shall only be active for one cycle only *)
IF in = FALSE THEN
	Q0 := FALSE;
	Q1 := FALSE;
	Q2:= FALSE;
	Q3 := FALSE;
END_IF;

IF in AND NOT edge AND NOT tx.q THEN
	(* a rising edge on in sets the counter to 0 *)
	cnt := 0;
ELSIF tx.Q AND NOT IN AND edge THEN
	(* count falling edges when tp.q is true *)
	cnt := cnt + 1;
ELSIF NOT tx.Q THEN
	CASE cnt OF
		0 : Q0 := TRUE;
		1 : Q1 := TRUE;
		2 : Q2 := TRUE;
		3 : Q3 := TRUE;
	END_CASE;
	cnt := -1;
END_IF;

(* remember the status of IN *)
edge := IN;
tx(in := IN, pt := TC);



(* revision history

hm 17. jul. 2008	rev 1.0
	original version released

hm	14. mar. 2009	rev 1.1
	removed double assignments

*)
END_FUNCTION_BLOCK
