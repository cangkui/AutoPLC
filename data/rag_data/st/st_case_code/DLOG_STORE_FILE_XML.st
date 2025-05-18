FUNCTION_BLOCK DLOG_STORE_FILE_XML
VAR_IN_OUT
    X : DLOG_DATA;
    SAVE_DATA : DLOG_SAVE;
END_VAR

VAR_INPUT
    ENABLE : BOOL;
    TRIG_M : BOOL;
    TRIG_T : TIME;
    FILENAME : STRING;
    DTI : DT;
    AUTO_CLOSE : TIME := TIME#15s0ms;
END_VAR

VAR_OUTPUT
    ERROR_C : DWORD;
    ERROR_T : BYTE;
END_VAR

VAR
    FS : FILE_SERVER;
    FSD : FILE_SERVER_DATA;
    PT : NETWORK_BUFFER;
    UCB : UNI_CIRCULAR_BUFFER;
    trig_auto : BOOL;
    trig_stored : BOOL;
    trig_tmp : BOOL;
    trig_m_last : BOOL;
    dti_last : DT := DATE_AND_TIME#2070-2-6-6:28:15;
    fn_last : STRING;
    fn : STRING;
    str : STRING[OSCAT_BASIC.STRING_LENGTH];
    step_1 : INT;
    step_2 : INT;
    idx : INT;
    log_state : USINT;
    log_stop : BOOL;
    n : INT;
    wd_ton : TON;
    wd_time : TIME;
    aw_ton : TON;
    aw_enable : BOOL;
    aw_aktiv : BOOL;
    total_bytes : UDINT;
END_VAR
X.DTI := DTI; (* Datum / Zeit eintragen *)
X.CLOCK_TRIG := DTI <> dti_last;
dti_last := DTI;
trig_auto := FALSE;
IF X.CLOCK_TRIG THEN (* Impuls in jeder Sekunde *)
	fn := OSCAT_BASIC.DT_TO_STRF(DTI, 0, FILENAME, 0); (* automatische Dateinamen generierung *)
	IF TRIG_T >= T#1000ms THEN (* automatischer Trigger *)
		trig_auto := DT_TO_UDINT(DTI) MOD (TIME_TO_UDINT(TRIG_T) / UDINT#1000) = UDINT#0;
	END_IF;
END_IF;

CASE step_1 OF

00:	IF ENABLE THEN
		aw_enable := AUTO_CLOSE >= T#5s;
		wd_time := SEL(X.LOAD_TIME_MAX > T#0s, T#5ms,X.LOAD_TIME_MAX);
		fn_last := fn; (* aktuellen Dateinnamen speichern *)
		IF X.ID_MAX = USINT#0 THEN (* nur einmal nach Satellitenbausteinen scannen *)
			X.ADD_COM := 01; (* ADD INFO *)
			X.STORE_TYPE := BYTE#3; (* XML-Modus *)
		END_IF;
		step_1 := 10;

		(* ---------------- Debug-Message -----------------------*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'FILE_XML1 - Start: File [~1], [~2]';
			LOG_CL.PRINTF[1] := fn_last;
			LOG_CL.PRINTF[2] := DT_TO_STRING(DTI);
			LOG_MSG();
		END_IF;*)
		(* -----------------------------------------------------------------*)

	END_IF;

10:	X.UCB.D_STRING := fn_last;
	IF fn_last = SAVE_DATA.FN_REM THEN
		X.UCB.D_HEAD := WORD#16#F201; (* Filemode open + write *)
		X.ADD_COM := 00;
	ELSE
		X.UCB.D_HEAD := WORD#16#F301; (* Filemode create + write *)
		X.ADD_COM := 02; (* ADD HEADER *)
	END_IF;

	X.UCB.D_MODE := 1; (* Dateinamen als String ablegen *)
	UCB(DATA:=X.UCB); (* Daten eintragen *)
	step_1 := 30;

30:	(* LOG-Trigger verarbeiten *)
	trig_tmp := (TRIG_M AND NOT trig_m_last) OR trig_auto OR X.ADD_DATA_REQ;

	IF ENABLE = FALSE OR fn <> fn_last THEN (* ENABLE = 0 oder neuen Dateinamen -> Log Stop *)
		trig_stored := trig_tmp; (* aktuellen Triggerpuls speichern *)
		X.UCB.D_HEAD := WORD#16#EE00;
		X.UCB.D_MODE := 1;
		UCB(DATA:=X.UCB); (* Element schreiben *)
		X.ADD_COM := 00;
		step_1 := 00;

		(* ---------------- Debug-Message -----------------------*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'FILE_XML1 - Stop: [~2]';
			LOG_CL.PRINTF[2] := DT_TO_STRING(DTI);
			LOG_MSG();
		END_IF;*)
		(* -----------------------------------------------------------------*)

	ELSIF ENABLE THEN
		IF trig_tmp OR trig_stored THEN (* manueller, automatischer oder externer Trigger *)
			trig_stored := FALSE;
			X.ADD_COM := 03; (* ADD DATA *)
			SAVE_DATA.TRIG_CNT := SAVE_DATA.TRIG_CNT + UDINT#1;
			SAVE_DATA.TRIG_CNT_TOTAL  := SAVE_DATA.TRIG_CNT_TOTAL + UDINT#1;
			(* ---------------- Debug-Message -----------------------*)
			(*IF _debug_enable THEN
				LOG_CL.NEW_MSG := 'FILE_XML1 - Trig: [~2]';
				LOG_CL.PRINTF[2] := DT_TO_STRING(DTI);
				LOG_MSG();
			END_IF;*)
			(* -----------------------------------------------------------------*)
		ELSE
			X.ADD_COM := 04;	(* ADD DATA REQ *)
		END_IF;
	END_IF;

END_CASE;
X.ADD_DATA_REQ := FALSE; (* externe Anforderung rücksetzen *)

(* ##################################### *)

CASE step_2 OF

00:	aw_ton(IN:= idx > 0 AND aw_enable , PT:= AUTO_CLOSE); (* AutoWrite Timer *)
	IF aw_ton.Q THEN (* Zeitgesteuertes Datenschreiben *)
		aw_aktiv := TRUE;
		step_2 := 10; (* Daten schreiben *)
		(* ---------------- Debug-Message -----------------------*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'FILE_XML2 - AutoWrite: [~2]';
			LOG_CL.PRINTF[2] := DT_TO_STRING(DTI);
			LOG_MSG();
		END_IF;*)
		(* -----------------------------------------------------------------*)

	ELSE
		wd_ton(IN:= FALSE, PT:= wd_time);
		wd_ton(IN:= TRUE);
		WHILE X.UCB.BUF_COUNT > 0 AND step_2 = 0 AND wd_ton.Q = FALSE DO

			X.UCB.D_MODE := 10;
			UCB(DATA:=X.UCB); (* Element lesen , aber noch nicht entfernen *)

			IF X.UCB.D_HEAD > WORD#16#F000 THEN (* F301 oder F201 *)

				str := '<?xml version="1.0" encoding="UTF-8"?><table>';

				IF idx + LEN(str) < SIZEOF(PT.BUFFER) THEN (* Platz für Element vorhanden ? *)

					FSD.MODE := WORD_TO_BYTE(SHR(SHL(X.UCB.D_HEAD,4),12)); (* 0X00 -> 000X = 3 create + write / 2 open + write *)
					FSD.OFFSET := UDINT#0; (* Start bei Dateianfang *)
					FSD.FILENAME := X.UCB.D_STRING; (* Dateiname *)
					PT.SIZE := UINT#0;
					SAVE_DATA.FN_REM := FSD.FILENAME; (* aktuellen Dateinamen remanent sichern *)

					X.NEW_FILE := FSD.FILENAME;
					X.NEW_FILE_SIZE := UDINT#0;
					X.NEW_FILE_RTRIG := FALSE;
					step_2 := 20;
	
					(* Headerdaten schreiben *)
					IF FSD.MODE = BYTE#3 THEN
						SAVE_DATA.TRIG_CNT := UDINT#0; (* Datei neu anlegen -> dann zähler reset *)

						OSCAT_BASIC._STRING_TO_BUFFER(STR:=str, POS:=idx, PT:=ADR(PT.BUFFER), SIZE:=SIZEOF(PT.BUFFER));
						idx := idx + LEN(str);
						X.NEW_FILE_SIZE := total_bytes + INT_TO_UDINT(idx);
					END_IF;

					X.UCB.D_MODE := 11;
					UCB(DATA:=X.UCB); (* Element entfernen *)

					(* ---------------- Debug-Message -----------------------*)
					(*IF _debug_enable THEN
						LOG_CL.NEW_MSG := 'FILE_XML2 - Create: Mode [~1] , [~2]';
						LOG_CL.PRINTF[2] := DT_TO_STRING(DTI);
						LOG_CL.PRINTF[1] := BYTE_TO_STRING(FSD.MODE);
						LOG_MSG();
					END_IF;*)
					(* -----------------------------------------------------------------*)
				END_IF;

			ELSIF X.UCB.D_HEAD = WORD#16#EE00 THEN (* LOG_STOP Command *)

				str := '</table>';

				IF idx + LEN(str) < SIZEOF(PT.BUFFER) THEN (* Platz für Element vorhanden ? *)

					OSCAT_BASIC._STRING_TO_BUFFER(STR:=str, POS:=idx, PT:=ADR(PT.BUFFER), SIZE:=SIZEOF(PT.BUFFER));
					idx := idx + LEN(str);
					log_stop := TRUE;
					step_2 := 10; (* Daten schreiben *)

					X.UCB.D_MODE := 11;
					UCB(DATA:=X.UCB); (* Element entfernen *)

				END_IF;

			ELSIF idx + LEN(X.UCB.D_STRING) + 120 < SIZEOF(PT.BUFFER) THEN (* Platz für Element vorhanden ? *)

				(* ---------------- Debug-Message -----------------------*)
				(*IF _debug_enable THEN
					LOG_CL.NEW_MSG := 'FILE_XML2 - BUF: TXT [~1],Pos [~2], [~3]';
					LOG_CL.PRINTF[1] := X.UCB.D_STRING;
					LOG_CL.PRINTF[2] := INT_TO_STRING(idx);
					LOG_CL.PRINTF[3] := DT_TO_STRING(DTI);
					LOG_MSG();
				END_IF;*)
				(* -----------------------------------------------------------------*)

				IF log_state = USINT#0 THEN
					str := '<row><entry>';
				ELSE
					str := '<entry>';
				END_IF;

				OSCAT_BASIC._STRING_TO_BUFFER(STR:=str, POS:=idx, PT:=ADR(PT.BUFFER), SIZE:=SIZEOF(PT.BUFFER));
				idx := idx + LEN(str);
				log_state := log_state + USINT#1;

				OSCAT_BASIC._STRING_TO_BUFFER(STR:=X.UCB.D_STRING, POS:=idx, PT:=ADR(PT.BUFFER), SIZE:=SIZEOF(PT.BUFFER));
				idx := idx + LEN(X.UCB.D_STRING);

				n := n + USINT#1;
				IF n >= X.ID_MAX THEN
					n := USINT#0;
					str := '</entry></row>';
					log_state := USINT#0;
				ELSE
					str := '</entry>';
				END_IF;
				OSCAT_BASIC._STRING_TO_BUFFER(STR:=str, POS:=idx, PT:=ADR(PT.BUFFER), SIZE:=SIZEOF(PT.BUFFER));
				idx := idx + LEN(str);
				X.NEW_FILE_SIZE := total_bytes + INT_TO_UDINT(idx);

				X.UCB.D_MODE := 11;
				UCB(DATA:=X.UCB); (* Element entfernen *)

			ELSE (* kein Platz vorhanden *)
				step_2 := 10; (* Daten schreiben *)
			END_IF;
			wd_ton();
		END_WHILE;
	END_IF;

10:	IF idx > 0 THEN (* Daten schreiben *)
		FSD.MODE := BYTE#2; (* File_Open_Exist + File_Write *)
		FSD.OFFSET := UDINT#4294967294; (* 16#FFFF_FFFE ->Spezialcode: Daten automatisch an Dateiende anhängen *)
		PT.SIZE := INT_TO_UINT(idx); (* Datenlänge eintragen *)
		idx := 0;
		(* ---------------- Debug-Message -----------------------*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'FILE_XML2 - WriteData: Mode [~3] Size [~1] , [~2]';
			LOG_CL.PRINTF[1] := UINT_TO_STRING(PT.SIZE);
			LOG_CL.PRINTF[2] := DT_TO_STRING(DTI);
			LOG_CL.PRINTF[3] := BYTE_TO_STRING(FSD.MODE);
			LOG_MSG();
		END_IF;*)
		(* -----------------------------------------------------------------*)
	END_IF;
	step_2 := 20;

20:	IF FSD.MODE = BYTE#0 THEN
		total_bytes := FSD.FILE_SIZE;
		X.NEW_FILE_SIZE := total_bytes + INT_TO_UDINT(idx);
		IF log_stop OR aw_aktiv THEN
			aw_aktiv := FALSE;
			FSD.MODE := BYTE#5; (* Datei schliessen *)
			(* ---------------- Debug-Message -----------------------*)
			(*IF _debug_enable THEN
				LOG_CL.NEW_MSG := 'FILE_XML2 - Close: [~2]';
				LOG_CL.PRINTF[2] := DT_TO_STRING(DTI);
				LOG_MSG();
			END_IF;*)
			(* -----------------------------------------------------------------*)
		END_IF;
		step_2 := 30;
	END_IF;

30:	IF FSD.MODE = BYTE#0 THEN (* Datei schliessen beendet *)
		IF log_stop THEN
			X.NEW_FILE_RTRIG := TRUE;
			log_stop := FALSE;
			SAVE_DATA.FN_REM := ''; (* remanenten Dateinamen löschen *)
		END_IF;
		step_2 := 00;
	END_IF;

END_CASE;

FS(FSD:=FSD,PT:=PT); (* FILE_SERVER *)

(* Fehlerauswertung *)
IF FSD.ERROR <> BYTE#0 THEN
	ERROR_C := BYTE_TO_DWORD(FSD.ERROR);
	ERROR_T := BYTE#1;
ELSIF X.UCB.BUF_DATA_LOST > UDINT#0 THEN
	ERROR_C := DWORD#2; (* Fehler: Bufferüberlauf - Datenmenge reduzieren *)
	ERROR_T := BYTE#6; 	(* Type: Buffer *)
ELSIF X.UCB.BUF_USED_MAX > USINT#90 THEN
	ERROR_C := DWORD#1; (* Warnung: Buffer auslastung > 90% *)
	ERROR_T := BYTE#6;	(* Type: Buffer *)
ELSE
	ERROR_C := DWORD#0;
	ERROR_T := BYTE#0;
END_IF;

trig_m_last := TRIG_M;

(* revision history
ks	25. july. 2011	rev 1.0
	original version

ks	16. apr. 2012	rev 1.1
	neuer überarbeiteter Code für automatisches File-open/close

ks	8. mar. 2016	rev 1.2
	nur einmal nach satellitenbausteinen scannen

ks	21. mar. 2016	rev 1.3
	prüfung auf freien bufferplatz hinzugefügt
	fehlermeldungen hinzugefügt

*)
END_FUNCTION_BLOCK
