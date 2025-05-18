FUNCTION_BLOCK TN_SC_VIEWPORT
VAR
    FB_TN_SC_WRITE : TN_SC_WRITE; 
    FB_TON : TON; 
    in_index : INT; 
    in_Color : INT; 
    by_Attr : BYTE; 
    in_count : INT; 
    Xus_LOG_VIEWPORT : us_LOG_VIEWPORT; 
    Xus_LOG_CONTROL : LOG_CONTROL; 
    Xus_TN_SCREEN : us_TN_SCREEN; 
    Iin_X : INT; 
    Iin_Y : INT; 
    Iin_Width : INT; 
    Idw_ATTR_1 : DWORD; 
    Idw_ATTR_2 : DWORD; 
    Iti_TIME : TIME; 
END_VAR
(* LOG-Messages ausgeben *)

IF Xus_LOG_VIEWPORT.UPDATE AND FB_TON.Q THEN (* Update gewünscht und Verzögerungszeit abgelaufen *)
	Xus_LOG_VIEWPORT.UPDATE := FALSE;
	FOR in_count := 1 TO Xus_LOG_VIEWPORT.COUNT DO (* Alle Viewport-Zeilen abarbeiten *)
		in_index := Xus_LOG_VIEWPORT.LINE_ARRAY[in_count];
		(* Farbe aus den Option-Parameter auslesen >0,Level,Back,Front< *)
		in_Color := DWORD_TO_INT(Xus_LOG_CONTROL.MSG_OPTION[in_index] AND DWORD#2#1111);

		CASE in_Color OF
		0..3: by_Attr := OSCAT_BASIC.BYTE_OF_DWORD(Idw_ATTR_1,INT_TO_BYTE(in_Color));
		4..7: by_Attr := OSCAT_BASIC.BYTE_OF_DWORD(Idw_ATTR_2,INT_TO_BYTE(in_Color));
		ELSE
			by_Attr := OSCAT_BASIC.BYTE_OF_DWORD(Idw_ATTR_1,BYTE#0);
		END_CASE;

		(* Message (Text + Farbe) ausgeben *)
		FB_TN_SC_WRITE(	Iin_Y := Iin_Y + in_count - INT#01,
						Iin_X := Iin_X,
						Iby_ATTR := by_Attr,
						Ist_STRING := OSCAT_BASIC.FIX(str := Xus_LOG_CONTROL.MSG[in_index], L := Iin_Width, C := BYTE#32, M := INT#00), (* String rechts ergänzen bzw. kürzen *)
						Xus_TN_SCREEN := Xus_TN_SCREEN);

	END_FOR;

	(* Timer stoppen/rücksetzen *)
	FB_TON.IN := FALSE;
	FB_TON.PT := Iti_TIME;
	FB_TON();

END_IF;

(* Update-Zeitverzoegerung *)
FB_TON.IN := TRUE;
FB_TON.PT := Iti_TIME;
FB_TON();

(* revision history
ks	10. mai. 2010	rev 1.0
	original version
*)
END_FUNCTION_BLOCK
