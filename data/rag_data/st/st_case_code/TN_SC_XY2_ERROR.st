FUNCTION TN_SC_XY2_ERROR : BOOL
VAR_INPUT
    X1 : INT;    
    Y1 : INT;    
    X2 : INT;    
    Y2 : INT;    
END_VAR
(* min und max werte prüfen *)
TN_SC_XY2_ERROR := Y1 < 0 OR X1 < 0 OR Y2 < 0 OR X2 < 0 OR Y1 > 23 OR X1 > 79 OR Y2 > 23 OR X2 > 79;

(* revision history
ks	10. mai. 2010	rev 1.0
	original version
*)
END_FUNCTION
