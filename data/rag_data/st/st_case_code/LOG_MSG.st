FUNCTION_BLOCK LOG_MSG
VAR
    FB_PRINT_SF : PRINT_SF;  // Instance of the derived type PRINT_SF
    idX : INT;               // Integer variable
END_VAR
IF DWORD_TO_BYTE(SHL(LOG_CL.NEW_MSG_OPTION,16)) > LOG_CL.LEVEL THEN RETURN; END_IF; (* Logging freigegeben ?*)

FB_PRINT_SF(PRINTF_DATA := LOG_CL.PRINTF,STR := LOG_CL.NEW_MSG);

IF LOG_CL.RESET THEN (* Logeinträge löschen *)
	LOG_CL.RESET := FALSE;
	LOG_CL.RING_MODE := FALSE;
	LOG_CL.IDX := 0;
	LOG_CL.UPDATE_COUNT := LOG_CL.UPDATE_COUNT + UINT#1; (* UpdateCount erhöhen *)
END_IF;

IF LOG_CL.SIZE > 0 THEN (* size wert ist eingetragen *)
	IF LEN(LOG_CL.NEW_MSG) > 0 THEN (* Meldung in Puffer eintragen *)
		IF (LOG_CL.IDX >= LOG_CL.SIZE) THEN
			LOG_CL.RING_MODE := TRUE; (* bei Überlauf = Ringbuffermodus *)
			LOG_CL.IDX := 0; (* wieder auf arrayanfang positionieren *)
		END_IF;

		LOG_CL.UPDATE_COUNT := LOG_CL.UPDATE_COUNT + UINT#1; (* UpdateCount erhöhen *)
		LOG_CL.IDX := LOG_CL.IDX + INT#1; (* nächste freie Position *)
		idx := LOG_CL.IDX; (* lade aktuelle Log-Puffer Positition *)
		LOG_CL.MSG[idx] := LOG_CL.NEW_MSG; (* neuen Log-String in puffer ablegen  *)
		LOG_CL.MSG_OPTION[idx] := LOG_CL.NEW_MSG_OPTION; (* neue Log-String Option in puffer ablegen  *)
		LOG_CL.NEW_MSG := ''; (* letzten Log-String löschen *)
	END_IF;
END_IF;
END_FUNCTION_BLOCK
