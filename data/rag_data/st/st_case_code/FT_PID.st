FUNCTION_BLOCK FT_PID
VAR_INPUT
    IN : REAL;
    KP : REAL := 1.0;
    TN : REAL := 1.0;
    TV : REAL := 1.0;
    ILIM_L : REAL := -1E+38;
    ILIM_H : REAL := 1E+38;
    IEN : BOOL := TRUE;
    RST : BOOL;
END_VAR

VAR_OUTPUT
    Y : REAL;
    LIM : BOOL;
END_VAR

VAR
    integ : FT_INT;
    diff : FT_DERIV;
END_VAR
(* run integrator only if TN > 0 *)
IF TN > 0.0 THEN
	integ(IN := IN, K := 1.0 / TN, RUN := IEN, RST := RST, OUT_MIN := ILIM_L, OUT_MAX := ILIM_H);
ELSE
	integ(RST := FALSE);
END_IF;

(* run differentiator *)
diff(IN := IN, K := TV);

(* combine both values *)
Y := KP * (integ.Out + diff.out + IN);

(* check if integrator has reached its limits and set overflow *)
LIM := integ.LIM;



(* revision history
hm 3. jun. 2008 	rev 2.0
	original version

hm	11. mar. 2009	rev 2.1
	real constants updated to new systax using dot

*)
END_FUNCTION_BLOCK
