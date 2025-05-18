FUNCTION_BLOCK DLOG_FILE_TO_SMTP
VAR_IN_OUT
    SERVER : STRING;
    MAILFROM : STRING;
    MAILTO : STRING[OSCAT_BASIC.STRING_LENGTH];
    SUBJECT : STRING;
    BODY : STRING[OSCAT_BASIC.STRING_LENGTH];
    X : DLOG_DATA;
END_VAR

VAR_INPUT
    FILE_DELETE : BOOL;
    TIMEOUT : TIME := TIME#30s0ms;
    DTI : DT;
    DTI_OFFSET : INT;
    RETRY : INT;
    RETRY_TIME : TIME := TIME#30s0ms;
    DNS_IP4 : DWORD;
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
    SMTP_CLIENT : SMTP_CLIENT;
    WT_1 : TON;
    ftrig_old : BOOL;
    files : STRING[OSCAT_BASIC.STRING_LENGTH];
    step : INT;
    smtp_busy : BOOL;
    smtp_done : BOOL;
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

00:	IF UCBD.BUF_COUNT > 0 AND smtp_busy = FALSE THEN
		UCBD.D_MODE := 12; (* Daten lesen und löschen *)
		UCB(DATA:=UCBD);
		files := UCBD.D_STRING;
		IF FILE_DELETE THEN
			files := CONCAT(files,';#DEL#');
		END_IF;
		cnt := RETRY;
		step := 10;

		(*####################  DEBUG-MESSAGE  ###################################*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'F_T_SMTP: Auftrag aktiv /file: ~2';
			LOG_CL.PRINTF[2] := files;
			LOG_MSG();
		END_IF;*)
		(*########################################################################*)

	END_IF;

10:	IF smtp_busy = FALSE THEN
		IF smtp_done THEN
			(*####################  DEBUG-MESSAGE  ###################################*)
			(*IF _debug_enable THEN
				LOG_CL.NEW_MSG := 'F_T_SMTP: Auftrag OK /file: ~2';
				LOG_CL.PRINTF[2] := files;
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
		IF RETRY = 0 THEN (* Transfer unbegrenzt wiederholen *)
			(*####################  DEBUG-MESSAGE  ###################################*)
			(*IF _debug_enable THEN
				LOG_CL.NEW_MSG := 'F_T_SMTP: Auftrag unbegrenzt wiederholen /file: ~2';
				LOG_CL.PRINTF[2] := files;
				LOG_MSG();
			END_IF;*)
			(*########################################################################*)
			step := 10; (* Transfer wiederholen *)
		ELSIF cnt >= 0 THEN
			(*####################  DEBUG-MESSAGE  ###################################*)
			(*IF _debug_enable THEN
				LOG_CL.NEW_MSG := 'F_T_SMTP: Auftrag wiederholen /cnt: ~1 /file: ~2';
				LOG_CL.PRINTF[1] := INT_TO_STRING(cnt);
				LOG_CL.PRINTF[2] := files;
				LOG_MSG();
			END_IF;*)
			(*########################################################################*)
			step := 10; (* Transfer wiederholen *)
		ELSE
			(*####################  DEBUG-MESSAGE  ###################################*)
			(*IF _debug_enable THEN
				LOG_CL.NEW_MSG := 'F_T_SMTP: Auftrag verworfen /file: ~2';
				LOG_CL.PRINTF[2] := files;
				LOG_MSG();
			END_IF;*)
			(*########################################################################*)
			step := 0; (* Auftrag verwerfen *)
		END_IF;
	END_IF;

END_CASE;

BUSY := step = 10;
DONE := step = 0;
WT_1(IN:= step = 20, PT:= RETRY_TIME);

SMTP_CLIENT(
	ACTIVATE:= BUSY,
	TIMEOUT:= TIMEOUT,
	DTI:= DTI,
	DTI_OFFSET:= DTI_OFFSET,
	DNS_IP4:= DNS_IP4,
	SERVER:= SERVER,
	MAILFROM:= MAILFROM,
	MAILTO:= MAILTO,
	SUBJECT:= SUBJECT,
	BODY:= BODY,
	FILES:= FILES,
	DONE=> smtp_done,
	BUSY=> smtp_busy
	);

(* Fehlerauswertung *)
IF SMTP_CLIENT.ERROR_T <> BYTE#0 THEN
	ERROR_C := SMTP_CLIENT.ERROR_C;
	ERROR_T := SMTP_CLIENT.ERROR_T;
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

ks	28. apr. 2011	rev 1.1
	stringfehler bei option 'file_delete'

ks	26. jul. 2011	rev 1.2
    codeoptimierung D_MODE 12

ks	21. mar. 2016	rev 1.3
	fehlermeldungen hinzugefügt
	fehler in RETRY korrigiert

*)
END_FUNCTION_BLOCK
