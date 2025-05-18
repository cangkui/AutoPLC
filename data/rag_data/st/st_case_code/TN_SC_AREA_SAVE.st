FUNCTION_BLOCK TN_SC_AREA_SAVE
VAR
    in_Offset_1 : INT;
    in_Offset_2 : INT;
    in_Y : INT;
    in_X : INT;
    
    Iin_Y1 : INT;
    Iin_X1 : INT;
    Iin_Y2 : INT;
    Iin_X2 : INT;
    
    Xus_TN_SCREEN : us_TN_SCREEN;
END_VAR
IF TN_SC_XY2_ERROR(Iin_X1,Iin_Y1,Iin_X2,Iin_Y2) THEN RETURN; END_IF;

(* Parameter ablegen *)
Xus_TN_SCREEN.bya_BACKUP[0] := BYTE#1;
Xus_TN_SCREEN.bya_BACKUP[1] := INT_TO_BYTE(Iin_X1);
Xus_TN_SCREEN.bya_BACKUP[2] := INT_TO_BYTE(Iin_Y1); 
Xus_TN_SCREEN.bya_BACKUP[3] := INT_TO_BYTE(Iin_X2);
Xus_TN_SCREEN.bya_BACKUP[4] := INT_TO_BYTE(Iin_Y2);

in_Offset_2 := 5;
(* CHAR und COLOR in BUFFER sichern *)
FOR in_Y := Iin_Y1 TO Iin_Y2 DO

	in_Offset_1 := (in_Y * 80) + Iin_X1;
	FOR in_X := Iin_X1 TO Iin_X2 DO
		Xus_TN_SCREEN.bya_BACKUP[in_Offset_2] := Xus_TN_SCREEN.bya_CHAR[in_Offset_1];
		in_Offset_2 := in_Offset_2 + INT#01;
		Xus_TN_SCREEN.bya_BACKUP[in_Offset_2] := Xus_TN_SCREEN.bya_COLOR[in_Offset_1];
		in_Offset_2 := in_Offset_2 + INT#01;
		in_Offset_1 := in_Offset_1 + INT#01;
	END_FOR;
END_FOR;

(* revision history
ks	10. mai. 2010	rev 1.0
	original version
*)
END_FUNCTION_BLOCK
