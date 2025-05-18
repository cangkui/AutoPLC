FUNCTION_BLOCK TN_SC_AREA_RESTORE
VAR
    in_Offset_1: INT; // Offset 1 input
    in_Offset_2: INT; // Offset 2 input
    in_Y: INT;        // Y coordinate input
    in_X: INT;        // X coordinate input
    in_Y1: INT;       // Y1 coordinate input
    in_X1: INT;       // X1 coordinate input
    in_Y2: INT;       // Y2 coordinate input
    in_X2: INT;       // X2 coordinate input
    Xus_TN_SCREEN: us_TN_SCREEN; // Output variable of derived type us_TN_SCREEN
END_VAR
IF Xus_TN_SCREEN.bya_BACKUP[0] > BYTE#00 THEN

	(* Parameter auslesen *)
	in_X1 := BYTE_TO_INT(Xus_TN_SCREEN.bya_BACKUP[1]);
	in_Y1 := BYTE_TO_INT(Xus_TN_SCREEN.bya_BACKUP[2]);
	in_X2 := BYTE_TO_INT(Xus_TN_SCREEN.bya_BACKUP[3]);
	in_Y2 := BYTE_TO_INT(Xus_TN_SCREEN.bya_BACKUP[4]);

	IF TN_SC_XY2_ERROR(in_X1,in_Y1,in_X2,in_Y2) THEN RETURN; END_IF;
	in_Offset_2 := 5;

	(* CHAR und COLOR von BUFFER wiederherstellen *)
	FOR in_Y := in_Y1 TO in_Y2 DO

		Xus_TN_SCREEN.bya_Line_Update[in_Y] := TRUE; (* Zeile zum Updaten freigeben *)
		in_Offset_1 := (in_Y * 80) + in_X1;

		FOR in_X := in_X1 TO in_X2 DO

			Xus_TN_SCREEN.bya_CHAR[in_Offset_1] := Xus_TN_SCREEN.bya_BACKUP[in_Offset_2];
			in_Offset_2 := in_Offset_2 + INT#01;

			Xus_TN_SCREEN.bya_COLOR[in_Offset_1] := Xus_TN_SCREEN.bya_BACKUP[in_Offset_2];
			in_Offset_2 := in_Offset_2 + INT#01;
			in_Offset_1 := in_Offset_1 + INT#01;

		END_FOR;
	END_FOR;

	(* Daten als ungültig markieren *)
	Xus_TN_SCREEN.bya_BACKUP[0] := BYTE#00;
END_IF;

(* revision history
ks	10. mai. 2010	rev 1.0
	original version
*)
END_FUNCTION_BLOCK
