FUNCTION_BLOCK GEN_RDT
VAR_INPUT
    Enable : BOOL := TRUE;
    Min_Time_ms : TIME := TIME#1s0ms; // min Taktzeit 
    Max_Time_ms : TIME := TIME#1s200ms; // Max Taktzeit 
    TP_Q : TIME := TIME#100ms; // Zeit Ausgang auf TRUE 
END_VAR

VAR_OUTPUT
    xQ : BOOL;
END_VAR

VAR
    tonRDMTimer : TON; // Zeitbaustein Taktgenerator 
    tof_xQ : TOF; // Ausschaltverzögerung Taktgenerator 
    tRDMTime : TIME; // Sollzeit 
    rRDMTime : REAL; // Zufalswert Timer 
END_VAR
tonRDMTimer ( IN:= Enable , PT:= tRDMTime );
tof_xQ ( IN:= tonRDMTimer.Q , PT:= TP_Q );
XQ := tof_xq.Q;
IF tonRDMTimer.Q THEN
		 xQ := TRUE ;
		 rRDMTime := RDM ( last:= rRDMTime ) ;
		 tRDMTime := REAL_TO_TIME ( rRDMTime * DINT_TO_REAL(TIME_TO_DINT( Max_Time_ms - Min_Time_ms ) + TIME_TO_DINT(Min_Time_ms ))) ;
		 tonRDMTimer ( IN:= FALSE );
END_IF;

(* revision history
J. Schohaus	19. nov 2007	rev 1.0
	origial version

hm	16. mar. 2008			rev 1.1
	added type conversion to avoid warnings under codesys 3.0
*)
END_FUNCTION_BLOCK
