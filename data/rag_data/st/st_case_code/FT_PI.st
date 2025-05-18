FUNCTION_BLOCK FT_PI
VAR_IN
    IN : REAL; 
    KP : REAL := 1.0; 
    KI : REAL := 1.0; 
    ILIM_L : REAL := -1E+38; 
    ILIM_H : REAL := 1E+38; 
    IEN : BOOL := TRUE; 
    RST : BOOL; 
END_VAR

VAR_OUT
    Y : REAL; 
    LIM : BOOL; 
END_VAR

VAR
    integ : FT_INT; 
END_VAR
(* run integrator *)
integ(IN := IN, K := KI, RUN := IEN, RST := RST, OUT_MIN := ILIM_L, OUT_MAX := ILIM_H);

(* check if integrator has reached its limits and set overflow *)
LIM := integ.LIM;
Y := KP * IN + integ.Out;



(* revision history
hm 	3. jun. 2008 	rev 2.0
	original version


*)
END_FUNCTION_BLOCK
