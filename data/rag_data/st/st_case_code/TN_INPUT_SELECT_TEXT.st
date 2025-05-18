FUNCTION_BLOCK TN_INPUT_SELECT_TEXT
VAR
    in_count : INT; 
    by_Attr : BYTE; 
    st_String : STRING; 
    FB_TN_SC_WRITE : TN_SC_WRITE;

    Xus_TN_SCREEN : us_TN_SCREEN; 
    Xus_TN_INPUT_CONTROL_DATA : us_TN_INPUT_CONTROL_DATA; 
END_VAR
(* Benutzereingabe auswerten -> Selektion Text n *)
IF Xus_TN_INPUT_CONTROL_DATA.bo_Focus = TRUE THEN

(* Leertaste-Taste auswerten *)
	IF Xus_TN_INPUT_CONTROL_DATA.by_Input_ASCII_Code = BYTE#32 THEN
		Xus_TN_INPUT_CONTROL_DATA.in_selected := Xus_TN_INPUT_CONTROL_DATA.in_selected + INT#01; 
		Xus_TN_INPUT_CONTROL_DATA.bo_Update_Input := TRUE;
	END_IF;
 
	(* Return-Taste auswerten *)
	IF Xus_TN_INPUT_CONTROL_DATA.by_Input_Exten_Code = BYTE#13 THEN
		Xus_TN_INPUT_CONTROL_DATA.bo_Input_Entered := TRUE;
	END_IF;

END_IF;

(* Aktualisierung der Anzeige *)
IF (Xus_TN_INPUT_CONTROL_DATA.bo_Update_Input = TRUE) OR (Xus_TN_INPUT_CONTROL_DATA.bo_Update_All = TRUE) THEN

	(* Title-Text ausgeben *)
	IF Xus_TN_INPUT_CONTROL_DATA.bo_Update_All = TRUE THEN

		IF (Xus_TN_INPUT_CONTROL_DATA.in_Title_X_Offset = INT#00) AND (Xus_TN_INPUT_CONTROL_DATA.in_Title_Y_Offset = INT#00) THEN
			(* wenn Offset X,Y =0 dann Titlestring links vom Element positionieren *)
			FB_TN_SC_WRITE.Iin_Y := Xus_TN_INPUT_CONTROL_DATA.in_Y;
			FB_TN_SC_WRITE.Iin_X := Xus_TN_INPUT_CONTROL_DATA.in_X - LEN(Xus_TN_INPUT_CONTROL_DATA.st_Title_String);

		ELSE
			(* Title-Text durch Row, Color-Offset ausgeben *)
			FB_TN_SC_WRITE.Iin_Y := Xus_TN_INPUT_CONTROL_DATA.in_Y + Xus_TN_INPUT_CONTROL_DATA.in_Title_Y_Offset;
			FB_TN_SC_WRITE.Iin_X := Xus_TN_INPUT_CONTROL_DATA.in_X + Xus_TN_INPUT_CONTROL_DATA.in_Title_X_Offset;
		END_IF;

		(* Title-Text durch Row, Color-Offset ausgeben *)
		FB_TN_SC_WRITE(	Iby_ATTR := Xus_TN_INPUT_CONTROL_DATA.by_Title_Attr,
						Ist_STRING := Xus_TN_INPUT_CONTROL_DATA.st_Title_String,
						Xus_TN_SCREEN := Xus_TN_SCREEN);

	END_IF;

	(* Eingabezeile verarbeiten *)

	(* Anzahl der Elemente bestimmen *)
	in_count := ELEMENT_COUNT(SEP:=BYTE#35,ELEMENT:=Xus_TN_INPUT_CONTROL_DATA.st_Input_Data);

	IF in_Count > INT#00 THEN

		IF Xus_TN_INPUT_CONTROL_DATA.in_selected < INT#1 THEN
			Xus_TN_INPUT_CONTROL_DATA.in_selected := INT#01;
		END_IF;

		IF Xus_TN_INPUT_CONTROL_DATA.in_selected > in_count THEN
			Xus_TN_INPUT_CONTROL_DATA.in_selected := INT#01;
		END_IF;

		st_String := ELEMENT_GET(SEP:=BYTE#35,POS:=Xus_TN_INPUT_CONTROL_DATA.in_selected - 1,ELEMENT:=Xus_TN_INPUT_CONTROL_DATA.st_Input_Data);

		st_String := OSCAT_BASIC.FIX(	str:=st_String,
							L:=LEN(Xus_TN_INPUT_CONTROL_DATA.st_Input_Mask),
							C:=BYTE#32,
							M:=Xus_TN_INPUT_CONTROL_DATA.in_Input_Option);

		Xus_TN_INPUT_CONTROL_DATA.st_Input_String := st_String;

		IF Xus_TN_INPUT_CONTROL_DATA.bo_Focus = TRUE THEN
			by_Attr := Xus_TN_INPUT_CONTROL_DATA.by_Attr_mF;
		ELSE
			by_Attr := Xus_TN_INPUT_CONTROL_DATA.by_Attr_oF;
		END_IF;

		FB_TN_SC_WRITE(	Iin_Y := Xus_TN_INPUT_CONTROL_DATA.in_Y,
						Iin_X := Xus_TN_INPUT_CONTROL_DATA.in_X,
						Iby_ATTR := by_Attr,
						Ist_STRING := st_String,
						Xus_TN_SCREEN := Xus_TN_SCREEN);
	END_IF;

	(* Cursor auf aktuelle Position setzen *)
	Xus_TN_INPUT_CONTROL_DATA.in_Cursor_Y := Xus_TN_INPUT_CONTROL_DATA.in_Y;
	Xus_TN_INPUT_CONTROL_DATA.in_Cursor_X := Xus_TN_INPUT_CONTROL_DATA.in_X;

	(* Update-Flag's rücksetzen *)
	Xus_TN_INPUT_CONTROL_DATA.bo_Update_All := FALSE;
	Xus_TN_INPUT_CONTROL_DATA.bo_Update_Input := FALSE;

END_IF;

(* revision history

ks	06. apr. 2011	rev 1.1
	list_len und list_get durch element_count und element_get ersetzt

ks	10. mai. 2010	rev 1.0
	original version
*)
END_FUNCTION_BLOCK
