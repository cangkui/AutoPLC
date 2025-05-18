FUNCTION_BLOCK DEAD_BAND_A
VAR_INPUT
    X : REAL;
    T : TIME;
    KL : REAL := 1.0;
    LM : REAL;
END_VAR

VAR_OUTPUT
    Y : REAL;
    L : REAL;
END_VAR

VAR
    tp1 : FT_PT1;
    tp2 : FT_PT1;
END_VAR
(* filter the input signal *)
tp1(in := X, T:= T);

(* filter the HF portion to generate a stable L *)
tp2(in := ABS(tp1.out - X), T := MULTIME(T, 4.0));

(* now we determine L which is half the width of the dead band *)
L := MIN(KL * tp2.out, LM);

IF X > L THEN
	Y := X - L;
ELSIF X < -L THEN
	Y := X + L;
ELSE
	Y := 0.0;
END_IF;


(* revision history
hm	14. jun. 2008	rev 1.0
	original version

hm	11. mar. 2009	rev 1.1
	real constants updated to new systax using dot

*)
END_FUNCTION_BLOCK
