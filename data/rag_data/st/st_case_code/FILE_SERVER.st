FUNCTION_BLOCK FILE_SERVER
VAR_IN_OUT
    FSD: FILE_SERVER_DATA;
    PT: NETWORK_BUFFER;
END_VAR

VAR
    buf_size: UDINT;
    handle: DWORD;
    read_max_length: UDINT;
    file_position: UDINT;
    used_filename: STRING;
    length_read: UDINT;
    write_length: UDINT;
    length_written: UDINT;
    open_handle: DWORD;
    seek_position: UDINT;
    tell_position: UDINT;
    file_size: DINT;
    open_mode: STRING[2];
    last_mode: BYTE;
    command: BYTE;
    error_code: BYTE;
    error: BOOL;
    file_change: BOOL;
    step: INT;
    FSR: FILE_SERVER_RUNTIME;
END_VAR
CASE step OF
0:
	IF FSD.MODE = LIMIT(1,FSD.MODE,5) THEN

		(* ---------------------- Debug-Message ----------------------------*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'File_Start Mode=~1';
			LOG_CL.PRINTF[1] := BYTE_TO_STRING(FSD.MODE);
			FB_LOG_MESSAGE();
		END_IF;*)
		(* -----------------------------------------------------------------*)

		buf_size := SIZEOF(PT.BUFFER);

		IF FSD.MODE = 4 THEN (* File_Remove *)
			IF handle > 0 THEN
				File_change := TRUE;
				step := 32000; (* File_Close ausführen *)
			ELSE
				step := 90; (* File_Remove ausführen *)
			END_IF;

		ELSIF FSD.MODE = 5 THEN (* File_Close *)
			step := 32000; (* File_Close ausführen *)

		ELSIF handle = 0 THEN (* MODE 1,2,3 *)
			step := 100; (* File muss erst geöffnet werden *)

		ELSE (* File ist geöffnset *)
			IF used_filename = FSD.FILENAME AND FSD.MODE = last_mode THEN (* kein Moduswechsel ? *)
				step := 200; (* ohne open prozedure weitermachen *)
			ELSE
				step := 32000;
				File_change := TRUE;
			END_IF;
		END_IF;
	END_IF;

(* ----------------------------------- *)
090: (* File_Remove *)
	command := 7; (* File Remove *)
	step := 95;

095: (* File_Remove beendet *)
	IF command = 0 THEN
		step := 30000; (* Beenden *)
	END_IF;
(* ----------------------------------- *)

100: (* File Size - Dateigroesse bestimmen *)
	command := 8; (* File Size *)
	step := 110;

110: (* File Size beendet *)
	IF command = 0 THEN
		IF FSD.MODE = 1 THEN (* open + read *)
			open_mode := 'r';
			FSD.FILE_SIZE := file_size; (* aktuelle Dateigroesse *)
		ELSIF FSD.MODE = 2 THEN (* open + write *)
			open_mode := 'a';
			FSD.FILE_SIZE := file_size; (* aktuelle Dateigroesse *)
		ELSE (* open new + write *)
			open_mode := 'w';
			FSD.FILE_SIZE := 0; (* aktuelle Dateigroesse *)
		END_IF;
		command := 1; (* File Open *)
		last_mode := FSD.MODE;
		step := 150; (* File_Open ausführen *)
	END_IF;

150: (* File Open *)
	IF command = BYTE#0 THEN
		IF error THEN
			step := 30000; (* Beenden *)
		ELSE
			(* Filedaten eintragen *)
			handle := open_handle;
			used_filename := FSD.FILENAME;
			file_position := UDINT#4294967295; (* maximal = unbekannt *)
			FSD.FILE_OPEN := TRUE;
			step := 200;
		END_IF;
	END_IF;

200: (* File Seek *)
	IF FSD.OFFSET = UDINT#4294967294 THEN (* 16#FFFF_FFFE -> Zeiger auf Dateiende stellen *)
		command := 5; (* File-Seek *)
		seek_position := FSD.FILE_SIZE;
		step := 210;
	ELSIF FSD.OFFSET > FSD.FILE_SIZE THEN (* prüfe auf gültige offset-vorgabe *)
		error_code := 255;
		step := 30000; (* Beenden *)
	ELSIF FSD.OFFSET <> file_position THEN
		command := 5; (* File-Seek *)
		seek_position := FSD.OFFSET;
		step := 210;
	ELSE
		step := 300;
	END_IF;

210: (* File Seek *)
	IF command = 0 THEN
		file_position := seek_position;
		IF error THEN
			step := 30000; (* Beenden *)
		ELSE
			step := 300;
		END_IF;
	END_IF;

300: (* Read or Write ? *)
	IF FSD.MODE = 1 THEN
		step := 400; (* Read *)
	ELSE
		step := 500; (* Write *)
	END_IF;

400: (* File Read *)
	IF UINT_TO_UDINT(PT.SIZE) > buf_size THEN (* Leselänge auf Buffergroesse begrenzen *)
		read_max_length := buf_size;
	ELSE
		read_max_length := UINT_TO_UDINT(PT.SIZE);
	END_IF;

	(* Leselänge an Dateilänge anpassen *)
	IF file_position + read_max_length > FSD.FILE_SIZE THEN
		read_max_length := FSD.FILE_SIZE - file_position;
	END_IF;

	IF read_max_length > 0 THEN
		command := 3; (* File_Read *)
		step := 410;
	ELSE
		PT.SIZE := UINT#0;
		step := 30000; (* Beenden *)
	END_IF;

410: (* File Read *)
	IF command = 0 THEN
		PT.SIZE := UDINT_TO_UINT(length_read);
		IF NOT Error THEN
			file_position := file_position + length_read; (* aktueller File_Position korrigieren *)
			FSD.OFFSET := file_position;
		END_IF;
		step := 30000; (* Beenden *)
	END_IF;

500: (* File Write *)
	IF UINT_TO_UDINT(PT.SIZE) > buf_size THEN (* Schreiblänge auf Buffergroesse begrenzen *)
		write_length := buf_size;
	ELSE
		write_length := UINT_TO_UDINT(PT.SIZE);
	END_IF;

	IF write_length > 0 THEN
		command := 4; (* File_Write *)
		step := 510;
	ELSE
		step := 30000; (* Beenden *)
	END_IF;

510:
	IF command = 0 THEN
		IF NOT Error THEN
			file_position := file_position + length_written; (* aktueller File_Position korrigieren *)
			FSD.OFFSET := file_position;
			IF file_position > FSD.FILE_SIZE THEN (* Datei ist groesser geworden *)
				FSD.FILE_SIZE := file_position;
			END_IF;
		END_IF;
		step := 30000; (* Beenden *)
	END_IF;

 (* --------------------------------------------- *)

30000: (* Befehl fertig *)
	FSD.MODE := 0;
	FSD.ERROR := error_code;
	step := 0;

	(* ---------------- Debug-Message -----------------------*)
	(*IF _debug_enable THEN
		LOG_CL.NEW_MSG := 'File_Stop: File ~1, error ~7, code ~8';
		LOG_CL.PRINTF[1]  := used_filename;
		LOG_CL.PRINTF[7]  := INT_TO_STRING(BOOL_TO_INT(error));
		LOG_CL.PRINTF[8]  := BYTE_TO_STRING(error_code);
		FB_LOG_MESSAGE();
	END_IF;*)
	(* --------------------------------------------- *)

32000:
	IF handle > 0 THEN
		command := 2; (* File_Close ausführen *)
	END_IF;
	step := 32100;

32100:
	IF command = 0 THEN (* File Close beendet *)
		(* Filedaten ablöschen *)
		FSD.FILE_OPEN := FALSE;
		file_position := 0;
		FSD.FILE_SIZE := 0;
		FSD.OFFSET := 0;
		used_filename := '';
		handle := 0;

		(* File_Name wurde fliegend geändert ? *)
		IF File_change THEN
			File_change := FALSE;
			step := 0;
		ELSE
			step := 30000;
		END_IF;
	END_IF;

END_CASE;

(* 1 = file_open
   2 = file_close
   3 = file_read
   4 = file_write
   5 = file_seek
   6 = file_tell
   7 = file_remove
*) 

FSR(MODE:=BYTE#0,COMMAND:=command,FSD:=FSD); (* File server Runtime / Befehlsstart überwachen *)

(* ------------ File Open ------------- *)
IF command = 1 THEN
	open_handle := SysFileOpen(FSD.FILENAME,open_mode);
	error := open_handle = 0;
	error_code := SEL(error,0,command);
	command := 0;
	FSR(MODE:=BYTE#1,COMMAND:=command,FSD:=FSD); (* File server Runtime / Befehlstop überwachen *)
	(* ---------------- Debug-Message -----------------------*)
	(*IF _debug_enable THEN
		LOG_CL.NEW_MSG := 'Open: File ~1, handle ~2 , mode ~3 , error ~7, code ~8';
		LOG_CL.PRINTF[1] := FSD.FILENAME;
		LOG_CL.PRINTF[2] := DWORD_TO_STRING(open_handle);
		LOG_CL.PRINTF[3] := open_mode;
		LOG_CL.PRINTF[7] := INT_TO_STRING(BOOL_TO_INT(error));
		LOG_CL.PRINTF[8] := BYTE_TO_STRING(error_code);
		FB_LOG_MESSAGE();
	END_IF;*)
	(* -----------------------------------------------------------------*)
END_IF;

(* ----------- File Close ------------- *)
IF command = 2 THEN
	error := NOT SysFileClose(handle);
	error_code := SEL(error,0,command);
	command := 0;
	FSR(MODE:=BYTE#1,COMMAND:=command,FSD:=FSD); (* File server Runtime / Befehlstop überwachen *)
	(* ---------------- Debug-Message -----------------------*)
	(*IF _debug_enable THEN
		LOG_CL.NEW_MSG := 'Close: File ~1, error ~7, code ~8';
		LOG_CL.PRINTF[1] := used_filename;
		LOG_CL.PRINTF[7] := INT_TO_STRING(BOOL_TO_INT(error));
		LOG_CL.PRINTF[8] := BYTE_TO_STRING(error_code);
		FB_LOG_MESSAGE();
	END_IF;*)
	(* -----------------------------------------------------------------*)
END_IF;

(* ------------ File Read ------------- *)
IF command = 3 THEN
	length_read := SysFileRead(handle, ADR(PT.BUFFER), read_max_length);
	error := length_read = 0;
	error_code := SEL(error,0,command);
	command := 0;
	FSR(MODE:=BYTE#1,COMMAND:=command,FSD:=FSD); (* File server Runtime / Befehlstop überwachen *)
	(* ---------------- Debug-Message -----------------------*)
	(*IF _debug_enable THEN
		LOG_CL.NEW_MSG := 'Read: File ~1, soll ~2,ist ~3 ,error ~7, code ~8';
		LOG_CL.PRINTF[1] := used_filename;
		LOG_CL.PRINTF[2] := UDINT_TO_STRING(read_max_length);
		LOG_CL.PRINTF[3] := UDINT_TO_STRING(length_read);
		LOG_CL.PRINTF[7] := INT_TO_STRING(BOOL_TO_INT(error));
		LOG_CL.PRINTF[8] := BYTE_TO_STRING(error_code);
		FB_LOG_MESSAGE();
	END_IF;*)
	(* -----------------------------------------------------------------*)
END_IF;

(* ------------ File Write ------------- *)
IF command = 4 THEN
	length_written := SysFileWrite(handle, ADR(PT.BUFFER), write_length);
	error := length_written = 0;
	error_code := SEL(error,0,command);
	command := 0;
	FSR(MODE:=BYTE#1,COMMAND:=command,FSD:=FSD); (* File server Runtime / Befehlstop überwachen *)
	(* ---------------- Debug-Message -----------------------*)
	(*IF _debug_enable THEN
		LOG_CL.NEW_MSG := 'Write: File ~1, soll ~2,ist ~3 ,error ~7, code ~8';
		LOG_CL.PRINTF[1] := used_filename;
		LOG_CL.PRINTF[2] := UDINT_TO_STRING(write_length);
		LOG_CL.PRINTF[3] := UDINT_TO_STRING(length_written);
		LOG_CL.PRINTF[7] := INT_TO_STRING(BOOL_TO_INT(error));
		LOG_CL.PRINTF[8] := BYTE_TO_STRING(error_code);
		FB_LOG_MESSAGE();
	END_IF;*)
	(* -----------------------------------------------------------------*)
END_IF;

(* ------------ File Seek ------------- *)
IF command = 5 THEN
	error := NOT SysFileSetPos(handle, seek_position);
	error_code := SEL(error,0,command);
	command := 0;
	FSR(MODE:=BYTE#1,COMMAND:=command,FSD:=FSD); (* File server Runtime / Befehlstop überwachen *)
	(* ---------------- Debug-Message -----------------------*)
	(*IF _debug_enable THEN
		LOG_CL.NEW_MSG := 'Seek: File ~1,pos ~3 ,error ~7, code ~8';
		LOG_CL.PRINTF[1] := used_filename;
		LOG_CL.PRINTF[3] := UDINT_TO_STRING(seek_position);
		LOG_CL.PRINTF[7] := INT_TO_STRING(BOOL_TO_INT(error));
		LOG_CL.PRINTF[8] := BYTE_TO_STRING(error_code);
		FB_LOG_MESSAGE();
	END_IF;*)
	(* -----------------------------------------------------------------*)
END_IF;

(* ------------ File Tell ------------- *)
IF command = 6 THEN
	tell_position := SysFileGetPos(handle);
	error := tell_position < 0;
	error_code := SEL(error,0,command);
	command := 0;
	FSR(MODE:=BYTE#1,COMMAND:=command,FSD:=FSD); (* File server Runtime / Befehlstop überwachen *)
	(* ---------------- Debug-Message -----------------------*)
	(*IF _debug_enable THEN
		LOG_CL.NEW_MSG := 'Tell: File ~1, pos ~3 ,error ~7, code ~8';
		LOG_CL.PRINTF[1] := used_filename;
		LOG_CL.PRINTF[3] := UDINT_TO_STRING(tell_position);
		LOG_CL.PRINTF[7] := INT_TO_STRING(BOOL_TO_INT(error));
		LOG_CL.PRINTF[8] := BYTE_TO_STRING(error_code);
		FB_LOG_MESSAGE();
	END_IF;*)
	(* -----------------------------------------------------------------*)
END_IF;

(* ------------ File Remove ------------- *)
IF command = 7 THEN
	error := NOT SysFileDelete(FSD.FILENAME);
	error_code := SEL(error,0,command);
	command := 0;
	FSR(MODE:=BYTE#1,COMMAND:=command,FSD:=FSD); (* File server Runtime / Befehlstop überwachen *)
	(* ---------------- Debug-Message -----------------------*)
	(*IF _debug_enable THEN
		LOG_CL.NEW_MSG := 'Remove: File ~1, error ~7, code ~8';
		LOG_CL.PRINTF[1] := FSD.FILENAME;
		LOG_CL.PRINTF[7] := INT_TO_STRING(BOOL_TO_INT(error));
		LOG_CL.PRINTF[8] := BYTE_TO_STRING(error_code);
		FB_LOG_MESSAGE();
	END_IF;*)
	(* -----------------------------------------------------------------*)
END_IF;

(* ------------ File Size ------------- *)
IF command = 8 THEN
	file_size := SysFileGetSize(FSD.FILENAME);
	error_code := 0; (* kein Errorcode vorhanden *)
	command := 0;
	FSR(MODE:=BYTE#1,COMMAND:=command,FSD:=FSD); (* File server Runtime / Befehlstop überwachen *)
	(* ---------------- Debug-Message -----------------------*)
	(*IF _debug_enable THEN
		LOG_CL.NEW_MSG := 'Size: File ~1, size ~3, error ~7, code ~8';
		LOG_CL.PRINTF[1] := FSD.FILENAME;
		LOG_CL.PRINTF[3] := DINT_TO_STRING(file_size);
		LOG_CL.PRINTF[7] := INT_TO_STRING(BOOL_TO_INT(error));
		LOG_CL.PRINTF[8] := BYTE_TO_STRING(error_code);
		FB_LOG_MESSAGE();
	END_IF;*)
	(* -----------------------------------------------------------------*)
END_IF;

(* ---------------- Debug-Message -----------------------*)
(*IF _debug_enable THEN
	IF step_last <> step THEN
		LOG_CL.NEW_MSG := 'Step: ~1';
		LOG_CL.PRINTF[1] := INT_TO_STRING(step);
		FB_LOG_MESSAGE();
	END_IF;
	step_last := step;
END_IF;*)
(* -----------------------------------------------------------------*)

(* revision history
ks	08. mar. 2016	rev 1.2
	Filesystem Laufzeitmessungen

ks	02. oct. 2014	rev 1.1.1
	PCWORX SRT File Delete error

ks	05. mai. 2012	rev 1.1
	auto_close funktion entfernt

ks	23. sep. 2010	rev 1.0
	original version
*)
END_FUNCTION_BLOCK
