FUNCTION_BLOCK TN_INPUT_CONTROL
// Input and Output Variables
VAR_IN_OUT
    Xus_TN_SCREEN : us_TN_SCREEN; // Derived type
    Xus_TN_INPUT_CONTROL : us_TN_INPUT_CONTROL; // Derived type
END_VAR

// Local Variables
VAR
    FB_TN_INPUT_SELECT_POPUP : TN_INPUT_SELECT_POPUP; // Derived type
    FB_TN_INPUT_SELECT_TEXT : TN_INPUT_SELECT_TEXT; // Derived type
    FB_TN_SC_WRITE_C : TN_SC_WRITE_C; // Derived type
    FB_TN_INPUT_EDIT_LINE : TN_INPUT_EDIT_LINE; // Derived type
    in_index : INT; // Integer type
    in_Scroll_Offset : INT; // Integer type
    in_Focus_new : INT; // Integer type
    in_Focus_alt : INT; // Integer type
    bo_Tool_Tip_Update : BOOL; // Boolean type
END_VAR
(* Bearbeitung freigegeben ? *)
IF (Xus_TN_INPUT_CONTROL.bo_Enable = FALSE) OR (Xus_TN_SCREEN.bo_Menue_Bar_Dialog = TRUE) THEN
	RETURN;
END_IF;

(* Fokus Reset durchführen *)
IF (Xus_TN_INPUT_CONTROL.bo_Reset_Fokus = TRUE) THEN
	Xus_TN_INPUT_CONTROL.bo_Reset_Fokus := FALSE;

	FOR in_index := INT#01 TO Xus_TN_INPUT_CONTROL.in_Count DO
		Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_index].bo_Focus := FALSE;
	END_FOR;

	IF (Xus_TN_INPUT_CONTROL.in_Count >= INT#01) THEN
		Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[1].bo_Focus := TRUE;
		Xus_TN_INPUT_CONTROL.in_Focus_at := INT#01;
	END_IF;

	Xus_TN_INPUT_CONTROL.bo_Update_all := TRUE;
END_IF;

IF (Xus_TN_SCREEN.bo_Modal_Dialog = FALSE) THEN
	(* TAB-Taste für Control-element wechsel auswerten *)
	IF (Xus_TN_SCREEN.by_Input_Exten_Code = BYTE#09) THEN

		in_Scroll_Offset := INT#01;

		(* Cursor-Oben Taste *)
	ELSIF (Xus_TN_SCREEN.by_Input_Exten_Code = BYTE#65) THEN

		in_Scroll_Offset := -1;

		(* Cursor-unten Taste *)
	ELSIF (Xus_TN_SCREEN.by_Input_Exten_Code = BYTE#66) THEN

		in_Scroll_Offset := INT#01;

	END_IF;
END_IF;

IF (in_Scroll_Offset <> INT#00) THEN  
	(* Grenzwerte überprüfen                                            *)

	Xus_TN_INPUT_CONTROL.in_Focus_at:=LIMIT(INT#01,Xus_TN_INPUT_CONTROL.in_Focus_at,Xus_TN_INPUT_CONTROL.in_Count);

	in_Focus_new := Xus_TN_INPUT_CONTROL.in_Focus_at + in_Scroll_Offset;

	IF (in_Focus_new > Xus_TN_INPUT_CONTROL.in_Count) THEN
		in_Focus_new := INT#1;
	END_IF;

	IF (in_Focus_new < INT#01) THEN
		in_Focus_new := Xus_TN_INPUT_CONTROL.in_Count;
	END_IF;

	in_Focus_alt := Xus_TN_INPUT_CONTROL.in_Focus_at;

	(* aktuellen Element den Fokus entziehen *)
	Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_Focus_alt].bo_Focus := FALSE;
	Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_Focus_alt].bo_Update_Input := TRUE;

	(* neuem Element den Fokus geben *)
	Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_Focus_new].bo_Focus  := TRUE;
	Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_Focus_new].bo_Update_Input := TRUE;

	Xus_TN_INPUT_CONTROL.in_Focus_at := in_Focus_new;

	bo_Tool_Tip_Update := TRUE;
	in_Scroll_Offset := INT#00;

END_IF;

(* Alle Elemente updaten ? *)
IF (Xus_TN_INPUT_CONTROL.bo_Update_all = TRUE) THEN
	FOR in_index := 1 TO Xus_TN_INPUT_CONTROL.in_Count DO
		Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_index].bo_Update_All := TRUE;
	END_FOR;
	Xus_TN_INPUT_CONTROL.bo_Update_all := FALSE;
	bo_Tool_Tip_Update := TRUE;
END_IF;

FOR in_index := 1 TO Xus_TN_INPUT_CONTROL.in_Count DO

	IF (Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_index].in_Type > INT#00) THEN

		IF (Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_index].bo_Focus = TRUE) THEN

			Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_index].by_Input_ASCII_Code  := Xus_TN_SCREEN.by_Input_ASCII_Code;
			Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_index].by_Input_Exten_Code  := Xus_TN_SCREEN.by_Input_Exten_Code;
			Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_index].bo_Input_ASCII_IsNum := Xus_TN_SCREEN.bo_Input_ASCII_IsNum;

			(* Tool-Tip Text ausgeben *)
			IF (bo_Tool_Tip_Update = TRUE) THEN

				FB_TN_SC_WRITE_C(	Iin_Y := Xus_TN_INPUT_CONTROL.in_ToolTip_Y,
									Iin_X := Xus_TN_INPUT_CONTROL.in_ToolTip_X,
									Iby_ATTR := Xus_TN_INPUT_CONTROL.by_ToolTip_Attr,
									Iin_OPTION := INT#02, (* Center-String *)
									Iin_LENGTH := Xus_TN_INPUT_CONTROL.in_ToolTip_Size,
									Ist_STRING := Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_index].st_Input_ToolTip,
									Xus_TN_SCREEN := Xus_TN_SCREEN);

				(* Flag rücksetzen *)
				bo_Tool_Tip_Update := FALSE;
			END_IF;
		END_IF;

		CASE Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_index].in_Type OF
		1:
			(* Edit-Line bearbeiten *)
			FB_TN_INPUT_EDIT_LINE(	Xus_TN_INPUT_CONTROL_DATA := Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_index],
									Xus_TN_SCREEN := Xus_TN_SCREEN);
		2:
			(* Select-Text bearbeiten *)
			FB_TN_INPUT_SELECT_TEXT(	Xus_TN_INPUT_CONTROL_DATA := Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_index],
										Xus_TN_SCREEN := Xus_TN_SCREEN);
		3:
			(* Select-Popup bearbeiten *)
			FB_TN_INPUT_SELECT_POPUP(	Xus_TN_INPUT_CONTROL_DATA := Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_index],
										Xus_TN_SCREEN := Xus_TN_SCREEN);
		END_CASE;

		IF (Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_index].bo_Focus = TRUE) THEN

			(* Cursor position von Element auf Screen übertragen *)
			Xus_TN_SCREEN.in_Cursor_X := Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_index].in_Cursor_X;
			Xus_TN_SCREEN.in_Cursor_Y := Xus_TN_INPUT_CONTROL.usa_TN_INPUT_CONTROL_DATA[in_index].in_Cursor_Y;

		END_IF;
	END_IF;
END_FOR;

(* revision history
ks	10. mai. 2010	rev 1.0
	original version
*)
END_FUNCTION_BLOCK
