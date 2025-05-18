FUNCTION INI_PARSER_FILE_DEMO
VAR
    STATION_NAME : STRING;
    STATION_IP : STRING;
    SMTP_SERVER : STRING;
    SMTP_FROM : STRING;
    IPF : INI_PARSER_FILE; // Assuming INI_PARSER_FILE is a previously defined data type
    FS : FILE_SERVER; // Assuming FILE_SERVER is a previously defined data type
    FSD : FILE_SERVER_DATA; // Assuming FILE_SERVER_DATA is a previously defined data type
    PT : NETWORK_BUFFER; // Assuming NETWORK_BUFFER is a previously defined data type
    section : INT;
    filename : STRING;
    key : STRING(OSCAT_BASIC.STRING_LENGTH);
    value : STRING(OSCAT_BASIC.STRING_LENGTH);
    str : STRING(OSCAT_BASIC.STRING_LENGTH);
    step : INT;
    result : BYTE;
    offset : UDINT;
    run : BYTE;
    done : BOOL;
END_VAR
(* Demo-Programm: INI-Datei mittels Enumeration auswerten *)

(* ----- den nachfolgenden Textblock als datei "config.ini" auf der sps ablegen ---
# Demo - Initialisierungsdatei

[Station]
NAME=Test_Station_1
IP=192.168.178.1

[SMTP]
SERVER=test.server.de
FROM=test1@gmx.net
---------------------------------------------------------------------------------*)

CASE step OF

00:	IF NOT done THEN
		filename := 'config.ini'; (* File_Daten einrichten *)
		OFFSET := 0; (* Start Offset vorgeben *)
		str := '';
		run := 3; (* key suchen *)
		step := 20;
	END_IF;

20:	IF run = 0 THEN (* kein Befehl mehr aktiv ? *)
		(* ---------------------- Debug-Message ----------------------------*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'IPFD: /key >~1< /value >~2< /offset ~3 /Run ~4 /Result ~5';
			LOG_CL.PRINTF[1] := key;
			LOG_CL.PRINTF[2] := value;
			LOG_CL.PRINTF[3] := UDINT_TO_STRING(OFFSET);
			LOG_CL.PRINTF[4] := BYTE_TO_STRING(run);
			LOG_CL.PRINTF[5] := BYTE_TO_STRING(result);
			LOG_MSG();
		END_IF;*)
		(* -----------------------------------------------------------------*)

		run := 3; (* vorbelegung -> naechsten key suchen *)

		IF result = 1 THEN (* neue Sektion gefunden *)
			IF key = 'Station' THEN
				section := 1;
			ELSIF key = 'SMTP' THEN
				section := 2;

			(* hier können weitere Section hinzugefügt werden *)

			ELSE
				section := 0; (* unbekannte sektion gefunden -> ignorieren *)
			END_IF;

		ELSIF result = 2 AND section > 0 THEN (* Key in einer bekannten sektion gefunden *)

			CASE section OF

			1:  (* Section: Station *)
				IF key = 'NAME' THEN
					STATION_NAME := value; (* Name *)
				ELSIF key = 'IP' THEN
					STATION_IP := value; (* IP-Adresse *)
				END_IF;

				(* hier können weitere Schlüsselsauswertungen hinzugefügt werden *)

			2:  (* Section: SMTP *)
				IF key = 'SERVER' THEN
					SMTP_SERVER := value; (* SMTP Server *)
				ELSIF key = 'FROM' THEN
					SMTP_FROM := value; (* SMTP Absender *)
				END_IF;

				(* hier können weitere Schlüsselsauswertungen hinzugefügt werden *)

			END_CASE;

		ELSIF result = 10 THEN (* nichts mehr gefunden *)
			run := 0; (* keine weitere INI Auswertung erwünscht *)
			FSD.MODE := 5; (* Datei schliessen *)
			step := 30;
		END_IF;
	END_IF;

30:	IF FSD.MODE = 0 THEN (* Befehl Datei schliessen beendet *)
		step := 0;
		done := TRUE;
	END_IF;

END_CASE;

IPF(FILENAME:=FILENAME,FSD:=FSD,STR:=str,RUN:=run,OFFSET:=OFFSET,KEY:=key,VALUE:=value,PT:=PT); (* INI PARAMETER FILE *)
result:=IPF.RESULT;

FS(FSD:=FSD,PT:=PT); (* FILE_SERVER *)
END_FUNCTION
