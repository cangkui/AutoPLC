FUNCTION_BLOCK CTRL_PWM
VAR_INPUT
    CI : REAL;
    MAN_IN : REAL;
    MANUAL : BOOL;
    F : REAL;
END_VAR

VAR_OUTPUT
    Q : BOOL;
END_VAR

VAR
    PW : PWM_DC;
END_VAR
PW(F := F, DC := SEL(MANUAL,CI,MAN_IN));
Q := PW.Q;


(* revision history
hm 3. jun. 2008 	rev 1.0
	original version

hm	21. oct. 2008	rev 1.1
	optimized code

*)
END_FUNCTION_BLOCK
