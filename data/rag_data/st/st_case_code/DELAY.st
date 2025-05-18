FUNCTION_BLOCK DELAY
VAR_INPUT
    IN : REAL;
    N : INT;
    RST : BOOL;
END_VAR

VAR_OUTPUT
    OUT : REAL;
END_VAR

VAR
    buf : ARRAY [0..31] OF REAL;
    i : INT;
    init : BOOL;
    stop : INT;
END_VAR
stop := LIMIT(0,N,32) - 1;
IF rst OR NOT init THEN
	init := TRUE;
	FOR i := 0 TO stop DO buf[i] := in; END_FOR;
	out := in;
	i := 0;
ELSIF stop < 0 THEN
	out := in;
ELSE
	out := buf[i];
	buf[i] := in;
	i := INC1(i, N);
END_IF;



(* revision history
hm 1.10.2006		rev 1.1
	corrected error in buffer management

hm 19.1.2007		rev 1.2
	changed reset to load the value of in instead of 0

hm	27. oct. 2008	rev 1.3
	improved performance

hm	23. feb.2009	rev 1.4
	corrected an index problem

hm	23. mar. 2009	rev 1.5
	corrected non standard write to input N

*)
END_FUNCTION_BLOCK
