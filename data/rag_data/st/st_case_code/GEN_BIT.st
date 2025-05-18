FUNCTION_BLOCK GEN_BIT
VAR_INPUT
    in0 : DWORD; 
    in1 : DWORD; 
    in2 : DWORD; 
    in3 : DWORD; 
    clk : BOOL; 
    steps : INT; 
    rep : INT; 
    rst : BOOL; 
END_VAR

VAR_OUTPUT
    Q0 : BOOL; 
    Q1 : BOOL; 
    Q2 : BOOL; 
    Q3 : BOOL; 
    cnt : INT; 
    run : BOOL; 
END_VAR

VAR
    r0 : DWORD; 
    r1 : DWORD; 
    r2 : DWORD; 
    r3 : DWORD; 
    rx : INT := 1; 
END_VAR
(* check if number of runs is finished or rep = 0 which means continuous *)

IF clk AND NOT rst THEN
	run := (rep = 0) OR (rx <= rep);
	IF run THEN
		(* check for step counter reached and reset to 0 if cnt = steps *)
		IF cnt = steps THEN
			cnt := 0;
		END_IF;

		(* when cnt = 0 then reload the inputs into the registers *)
		IF cnt = 0 THEN
			r0 := in0;
			r1 := in1;
			r2 := in2;
			r3 := in3;
		END_IF;

		(* when cnt < steps, shift the lowest bits to the outputs *)
		IF (cnt < steps) THEN
			Q0 := r0.0;
			Q1 := r1.0;
			Q2 := r2.0;
			Q3 := r3.0;
			r0 := SHR(r0,1);
			r1 := SHR(r1,1);
			r2 := SHR(r2,1);
			r3 := SHR(r3,1);
		END_IF;

		(* increment the step counter *)
		cnt := cnt +1;
		IF (cnt = steps) AND (rep <> 0) THEN rx := rx +1; END_IF;
		IF (rx > rep) AND (rep <> 0) THEN run := FALSE; END_IF;
	END_IF;
ELSE
	IF rst THEN
		run := FALSE;
		Q0 := FALSE;
		Q1 := FALSE;
		Q2 := FALSE;
		Q3 := FALSE;
		r0 := 0;
		r1 := 0;
		r2 := 0;
		r3 := 0;
		cnt := 0;
		rx := 1;
	END_IF;
END_IF;



(* revision histroy
hm	4 aug 2006	rev 1.0
	original version

hm 15. oct. 2008	rev 1.1
	improved performance

hm	14. mar. 2009	rev 1.2
	removed double assignments

*)
END_FUNCTION_BLOCK
