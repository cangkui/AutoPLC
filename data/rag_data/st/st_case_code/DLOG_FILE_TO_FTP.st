FUNCTION_BLOCK DLOG_FILE_TO_FTP
VAR_IN_OUT
    X : DLOG_DATA;
END_VAR

VAR_INPUT
    FTP_URL : ARRAY[0..OSCAT_BASIC.STRING_LENGTH-1] OF CHAR;
    FTP_ACTIV : BOOL;
    FILE_DELETE : BOOL;
    TIMEOUT : TIME := TIME#30s0ms;
    RETRY : INT;
    RETRY_TIME : TIME := TIME#30s0ms;
    DNS_IP4 : DWORD;
    PLC_IP4 : DWORD;
END_VAR

VAR_OUTPUT
    DONE : BOOL;
    BUSY : BOOL;
    ERROR_C : DWORD;
    ERROR_T : BYTE;
END_VAR

VAR
    UCBD : UNI_CIRCULAR_BUFFER_DATA;
    UCB : UNI_CIRCULAR_BUFFER;
    FTP_CLIENT : FTP_CLIENT;
    WT_1 : TON;
    ftrig_old : BOOL;
    filename : STRING;
    step : INT;
    ftp_busy : BOOL;
    ftp_done : BOOL;
    cnt : INT;
END_VAR
(* neue Filenamen speichern *)
IF X.NEW_FILE_RTRIG AND ftrig_old = FALSE THEN
	UCBD.D_HEAD := 1;
	UCBD.D_MODE := 1;
	UCBD.D_STRING := X.NEW_FILE;
	UCB(DATA:=UCBD); (* Daten eintragen *)
END_IF;
ftrig_old := X.NEW_FILE_RTRIG;

CASE step OF

00:	IF UCBD.BUF_COUNT > 0 AND ftp_busy = FALSE THEN
		UCBD.D_MODE := 12; (* Daten lesen und löschen *)
		UCB(DATA:=UCBD);
		filename := UCBD.D_STRING;
		cnt := RETRY;
		step := 10;

		(*####################  DEBUG-MESSAGE  ###################################*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'F_T_FTP: Auftrag aktiv /file: ~2';
			LOG_CL.PRINTF[2] := files;
			LOG_MSG();
		END_IF;*)
		(*########################################################################*)

	END_IF;

10:	IF ftp_busy = FALSE THEN
		IF ftp_done THEN
			(*####################  DEBUG-MESSAGE  ###################################*)
			(*IF _debug_enable THEN
				LOG_CL.NEW_MSG := 'F_T_FTP: Auftrag OK /file: ~2';
				LOG_CL.PRINTF[2] := filename;
				LOG_MSG();
			END_IF;*)
			(*########################################################################*)
			step := 0;
		ELSE
			step := 20; (* pause nach error *)
		END_IF;
	END_IF;

20:	IF WT_1.Q THEN (* Wartezeit *)
		cnt := cnt - 1;
		IF RETRY = 0 THEN (* FTP Transfer unbegrenzt wiederholen *)
			(*####################  DEBUG-MESSAGE  ###################################*)
			(*IF _debug_enable THEN
				LOG_CL.NEW_MSG := 'F_T_FTP: Auftrag unbegrenzt wiederholen /file: ~2';
				LOG_CL.PRINTF[2] := filename;
				LOG_MSG();
			END_IF;*)
			(*########################################################################*)
			step := 10; (* FTP Transfer wiederholen *)
		ELSIF cnt >= 0 THEN
			(*####################  DEBUG-MESSAGE  ###################################*)
			(*IF _debug_enable THEN
				LOG_CL.NEW_MSG := 'F_T_FTP: Auftrag wiederholen /cnt: ~1 /file: ~2';
				LOG_CL.PRINTF[1] := INT_TO_STRING(cnt);
				LOG_CL.PRINTF[2] := filename;
				LOG_MSG();
			END_IF;*)
			(*########################################################################*)
			step := 10; (* FTP Transfer wiederholen *)
		ELSE
			(*####################  DEBUG-MESSAGE  ###################################*)
			(*IF _debug_enable THEN
				LOG_CL.NEW_MSG := 'F_T_FTP: Auftrag verworfen /file: ~2';
				LOG_CL.PRINTF[2] := filename;
				LOG_MSG();
			END_IF;*)
			(*########################################################################*)
			step := 0; (* FTP Auftrag verwerfen *)
		END_IF;
	END_IF;

END_CASE;

BUSY := step = 10;
DONE := step = 0;
WT_1(IN:= step = 20, PT:= RETRY_TIME);

FTP_CLIENT(
	ACTIVATE:= BUSY,
	FILENAME:= filename,
	FTP_URL:= FTP_URL,
	FTP_DOWNLOAD:= FALSE,
	FTP_ACTIV:= FTP_ACTIV,
	FILE_DELETE:= FILE_DELETE,
	TIMEOUT:= TIMEOUT,
	DNS_IP4:= DNS_IP4,
	PLC_IP4:= PLC_IP4,
	DONE=> ftp_done,
	BUSY=> ftp_busy
	);

(* Fehlerauswertung *)
IF FTP_CLIENT.ERROR_T <> BYTE#0 THEN
	ERROR_C := FTP_CLIENT.ERROR_C;
	ERROR_T := FTP_CLIENT.ERROR_T;
ELSIF UCBD.BUF_DATA_LOST > UDINT#0 THEN
	ERROR_C := DWORD#2; (* Fehler: Bufferüberlauf - Datenmenge reduzieren *)
	ERROR_T := BYTE#6; 	(* Type: Buffer *)
ELSIF UCBD.BUF_USED_MAX > USINT#90 THEN
	ERROR_C := DWORD#1; (* Warnung: Buffer auslastung > 90% *)
	ERROR_T := BYTE#6;	(* Type: Buffer *)
ELSE
	ERROR_C := DWORD#0;
	ERROR_T := BYTE#0;
END_IF;

(* revision history
ks	01. jan. 2011	rev 1.0
	original version

ks	26. jul. 2011	rev 1.1
    codeoptimierung D_MODE 12

ks	21. mar. 2016	rev 1.2
	fehlermeldungen hinzugefügt
	fehler in RETRY korrigiert
	
*)
END_FUNCTION_BLOCK
