FUNCTION PERIOD2 : BOOL
VAR_INPUT
    DP : ARRAY[0..3, 0..1] OF DATE; // Array of DATEs
    DX : DATE;                     // Single DATE
END_VAR
PERIOD2 := 	(DX >= DP[0,0] AND DX <= DP[0,1]) OR
			(DX >= DP[1,0] AND DX <= DP[1,1]) OR
			(DX >= DP[2,0] AND DX <= DP[2,1]) OR
			(DX >= DP[3,0] AND DX <= DP[3,1]);


(* revision history

hm		27. apr 2008	rev 1.0
	original version


*)
END_FUNCTION
