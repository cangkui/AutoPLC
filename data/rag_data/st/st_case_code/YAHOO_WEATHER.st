FUNCTION_BLOCK YAHOO_WEATHER
VAR_IN_OUT
    IP_C : IP_C; // IP_Control Verwaltungsstruktur
    S_BUF : NETWORK_BUFFER;
    R_BUF : NETWORK_BUFFER;
    YW : YAHOO_WEATHER_DATA;
END_VAR

VAR_INPUT
    ACTIVATE : BOOL;
    UNITS : BOOL;
    LOCATION : STRING[20];
END_VAR

VAR_OUTPUT
    BUSY : BOOL;
    DONE : BOOL;
    ERROR_C : DWORD;
    ERROR_T : BYTE;
END_VAR

VAR
    CTRL : XML_CONTROL;
    XML_READER : XML_READER;
    URL_DATA : url;
    DNS_CLIENT : DNS_CLIENT;
    HTTP_GET : HTTP_GET;
    activate_last : BOOL;
    value_int : INT;
    value_real : REAL;
    v_real : REAL;
    state : INT;
    cycle : INT;
    day_idx : INT;
    idx : INT;
    cnt : INT;
    repeat_request : BOOL;
    repeat_wait_time : UDINT := 5000; // Repeat-Waittime
    repeat_count : USINT;
    repeat_max_count : USINT := 3; // maximal repeat
    repeat_tx : UDINT;
    tx : UDINT;
END_VAR
(* read system time *)
tx := OSCAT_BASIC.T_PLC_MS();

CASE state OF

00:	IF ACTIVATE AND NOT activate_last THEN
		state := 40;
		DONE := FALSE;
		BUSY := TRUE;
		ERROR_C := DWORD#0;
		ERROR_T := BYTE#0;
		cycle := 0;
		repeat_count := USINT#0;
		repeat_request := FALSE;
  		URL_DATA.DOMAIN := 'query.yahooapis.com'; (* Domainname für DNS_CLIENT *)
	END_IF;

40:	IF DNS_CLIENT.DONE THEN
		state := 50;
	ELSIF (DNS_CLIENT.ERROR > DWORD#00) THEN
		(* Fehlerbehandlung *)
		ERROR_C := DNS_CLIENT.ERROR;
		ERROR_T := BYTE#01;
		state   := 100;
	END_IF;

50:   	(* Teil 1: http://query.yahooapis.com/v1/public/yql?q=select%20units,wind,atmosphere,astronomy,location%20from%20weather.forecast%20where%20woeid=551801%20AND%20u=%27c%27
             Teil 2: http://query.yahooapis.com/v1/public/yql?q=select%20item%20from%20weather.forecast%20where%20woeid=551801%20AND%20u=%27c%27 *)

	IF cycle = 0 THEN
  		URL_DATA.QUERY := 'q=select%20units,wind,atmosphere,astronomy,location%20from%20weather.forecast%20where%20woeid='; (* Teil 1 - infos *)
  	ELSE
  		URL_DATA.QUERY := 'q=select%20item%20from%20weather.forecast%20where%20woeid='; (* Teil 2 - Tagesdaten *)
  	END_IF;

  	URL_DATA.QUERY := CONCAT(URL_DATA.QUERY,LOCATION); (* location *)

  	IF UNITS THEN
    		URL_DATA.QUERY := CONCAT(URL_DATA.QUERY,'%20and%20u=%27f%27'); (* fahrenheit *)
  	ELSE
    		URL_DATA.QUERY := CONCAT(URL_DATA.QUERY,'%20and%20u=%27c%27'); (* celsius *)
  	END_IF;

  	URL_DATA.PATH := '/v1/public/yql';

  	state := 60;

60:	IF HTTP_GET.DONE THEN
		state := 80;
		CTRL.START_POS := HTTP_GET.BODY_START;
		CTRL.STOP_POS := HTTP_GET.BODY_STOP;
		CTRL.COMMAND := WORD#2#10000000_00011000; (* ONLY TEXT AND ATTRIBUTE *)
		CTRL.WATCHDOG  := T#1ms;

	ELSIF (HTTP_GET.ERROR > DWORD#00) THEN
		(* Fehlerbehandlung *)
		ERROR_C := HTTP_GET.ERROR;
		ERROR_T := BYTE#02;
		state := 100;
	END_IF;

80: XML_READER(CTRL:=CTRL,BUF:=R_BUF.BUFFER); (* XML Daten seriell lesen *)
	IF CTRL.TYP < 98 THEN (* nur auswerten wenn kein Watchdog durchlauf *)
		value_int := 0;
		value_real := 0.0;
		(* automatische konvertierung in real und int ausführen *)
		IF LEN(CTRL.VALUE) <= 20 THEN
			v_real:=OSCAT_BASIC.FLOAT_TO_REAL(FLT:=CTRL.VALUE);
			IF OSCAT_BASIC.CHK_REAL(v_real) = 0 THEN (* !!! prüft auf gültigen REAL Wert, ansonsten kann Twincat 2.x abstürzen !!!!  *)
				value_real := v_real;
				value_int := REAL_TO_INT(value_real);
			END_IF;
		END_IF;

		(*####################  DEBUG-MESSAGE  ###################################*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'XML: ~2 # ~3 # ~4 # ~5';
			LOG_CL.PRINTF[2] := UINT_TO_STRING(CTRL.LEVEL);
			LOG_CL.PRINTF[3] := UINT_TO_STRING(CTRL.COUNT);
			LOG_CL.PRINTF[4] := CTRL.ELEMENT;
			LOG_CL.PRINTF[5] := CTRL.VALUE;
			LOG_MSG();
		END_IF;*)
		(*########################################################################*)

		(* Antwortdaten stehen noch nicht bereit -> Warten und Abfrage wiederholen *)
		IF ctrl.COUNT = 7 AND (ctrl.VALUE = '0')  THEN

			repeat_count := repeat_count + USINT#1;

			(* maximale Anzahl an Wiederholungen erreicht -> Mit Fehler abbrechen *)
			IF repeat_count > repeat_max_count THEN

				(* Fehlerbehandlung *)
				ERROR_C := USINT_TO_DWORD(repeat_max_count);
				ERROR_T := BYTE#03;
				state   := 100;

			ELSE
				repeat_request := TRUE;
				state := 100;

				(*####################  DEBUG-MESSAGE  ###################################*)
				(*IF _debug_enable THEN
					LOG_CL.NEW_MSG := 'Request Error  / Part ~2 / Count ~3';
					LOG_CL.PRINTF[2] := INT_TO_STRING(cycle);
					LOG_CL.PRINTF[3] := USINT_TO_STRING(repeat_count);
					LOG_MSG();
				END_IF;*)
				(*########################################################################*)

			END_IF;

		END_IF;

		IF cycle = 0 THEN
	      		(* ------------ Teil 1 --------------*)
	      		CASE UINT_TO_INT(CTRL.COUNT) OF
	        		014: YW.location_city := CTRL.VALUE;
	        		015: YW.location_country := CTRL.VALUE;
	        		016: YW.location_region := CTRL.VALUE;
	        		020: YW.wind_chill := value_int;
	        		021: YW.wind_direction := value_int;
	        		022: YW.wind_speed := value_real;
	        		026: YW.atmosphere_humidity := value_int;
	        		027: YW.atmosphere_pressure := value_real;
	        		028: YW.atmosphere_rising := value_int;
	        		029: YW.atmosphere_visibility := value_real;
	        		033: YW.astronomy_sunrise := CTRL.VALUE;
	        		034: YW.astronomy_sunset := CTRL.VALUE;
	        		038: YW.unit_distance := CTRL.VALUE;
	        		039: YW.unit_pressure := CTRL.VALUE;
	        		040: YW.unit_speed := CTRL.VALUE;
	        		041: YW.unit_temperature := CTRL.VALUE;
	      		END_CASE;
	    	ELSE
	      		(* ------------ Teil 2 --------------*)
	      		CASE UINT_TO_INT(CTRL.COUNT) OF
	        		018: YW.GEO_LATITUDE := value_real;
	        		022: YW.GEO_LONGITUDE := value_real;
	
			032: YW.CUR_CONDITIONS_CODE := value_int;
			         YW.CUR_CONDITIONS_ICON := value_int;
	        		034: YW.CUR_CONDITIONS_TEMP := value_int;
	        		035: YW.CUR_CONDITIONS_TEXT := CTRL.VALUE;
	
			039: YW.FORECAST_TODAY_CODE := value_int;
			         YW.FORECAST_TODAY_ICON := value_int;
			040: YW.FORECAST_TODAY_DATE_LONG := CTRL.VALUE;
	        		042: YW.FORECAST_TODAY_HIGH_TEMP := value_int;
	        		043: YW.FORECAST_TODAY_LOW_TEMP := value_int;
	        		044: YW.FORECAST_TODAY_TEXT := CTRL.VALUE;

			048..125: (* forcast day 1-10 *)
				cnt := UINT_TO_INT(CTRL.COUNT) - 48;
				day_idx := cnt / 9 + 1;
				idx := cnt MOD 9;
	      	  		CASE idx OF
					0:	YW.FORECAST_DAY[day_idx].CODE := value_int;
						YW.FORECAST_DAY[day_idx].ICON := value_int;
					1:	YW.FORECAST_DAY[day_idx].DATE_LONG := CTRL.VALUE;
					3:	YW.FORECAST_DAY[day_idx].HIGH_TEMP := value_int;
					4:	YW.FORECAST_DAY[day_idx].LOW_TEMP := value_int;
					5:	YW.FORECAST_DAY[day_idx].TEXT := CTRL.VALUE;
	      	  		END_CASE;
	      		END_CASE;
	    	END_IF;
	ELSIF CTRL.TYP = 99 THEN (* EXIT - letztes Element gelesen *)
		state := 100;
	END_IF;

100:	(* UNLOCK HTTP DATA *)
	IF (NOT HTTP_GET.DONE) THEN

		IF repeat_request THEN
			repeat_tx := tx;
    			state := 110;

		ELSIF cycle = 0 AND ERROR_T = BYTE#0 THEN
			(* Teil 2 ausführen *)
    			state := 50;
			cycle := cycle + 1;
			repeat_count := USINT#0;
		ELSE
	    		state := 0;
	    		BUSY  := FALSE;
	    		DONE := ERROR_T = BYTE#0;
		END_IF;
	END_IF;

110: (* REPEAT REQUEST - WAIT TIME *)
	IF tx > repeat_tx + repeat_wait_time THEN

		(*####################  DEBUG-MESSAGE  ###################################*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'Repeat request after waittime / Part ~2 / Count ~3';
			LOG_CL.PRINTF[2] := INT_TO_STRING(cycle);
			LOG_CL.PRINTF[3] := USINT_TO_STRING(repeat_count);
			LOG_MSG();
		END_IF;*)
		(*########################################################################*)

		repeat_request := FALSE;
		state := 50;

	END_IF;

END_CASE;

(* ---------------- DNS --------------- *)
DNS_CLIENT(	IP_C:=IP_C,
			S_BUF:=S_BUF,
			R_BUF:=R_BUF,
			DOMAIN:=URL_DATA.DOMAIN,
			IP4_DNS:=DWORD#00,
			ACTIVATE:=state=40 (* DNS Start *)
			);
(* nicht oder nicht direkt verwendete Parameter *)
(*	DWORD :=DNS_CLIENT.IP4;
	BOOL  :=DNS_CLIENT.DONE;
	BYTE  :=DNS_CLIENT.ERROR; *)
(* -------------------------------------*)

(* ------------- HTTP_GET --------------- *)
HTTP_GET(	IP_C:=IP_C,
			S_BUF:=S_BUF,
			R_BUF:=R_BUF,
			IP4:=DNS_CLIENT.IP4,
			GET:=state=60,
			MODE:=BYTE#2, (* HTTP 1.1 mit persistenter Verbindung *)
			UNLOCK_BUF:=state=100,
			URL_DATA:=URL_DATA
			);

(* nicht oder nicht direkt verwendete Parameter *)
(*	STRING := HTTP_GET.HTTP_STATUS;
	UINT :=HTTP_GET.HEADER_START;
	UINT :=HTTP_GET.HEADER_STOP;
	UINT :=HTTP_GET.BODY_START;
	UINT :=HTTP_GET.BODY_STOP;
	BOOL :=HTTP_GET.DONE;
	BYTE :=HTTP_GET.ERROR;
*)
(* -------------------------------------*)

activate_last := ACTIVATE;

(*####################  DEBUG-MESSAGE  ###################################*)
(*IF _debug_enable THEN
	IF state <> _debug_state_last THEN
		LOG_CL.NEW_MSG := 'yw-step: /last ~1 /New ~2';
		LOG_CL.PRINTF[1] := INT_TO_STRING(_debug_state_last);
		LOG_CL.PRINTF[2] := INT_TO_STRING(state);
		LOG_MSG();
		_debug_state_last := state;
	END_IF;
END_IF;*)
(*########################################################################*)

(* revision history
ks	18. dez. 2008	rev 1.0
	original version

ks	01. oct. 2009	rev 1.1
	YAHOO_WEATHER_DATA

ks	24. sep. 2010	rev 1.2
	new URL-Location-parameter and REAL_CHECK (TwinCat BUG) 

ks	04. may. 2011	rev 1.3
	neue elemente:  CUR_CONDITIONS_ICON, FORCAST_TODAY_ICON, FORCAST_TOMORROW_ICON

ks	30. mar. 2016	rev 1.4
	änderung der api schnittstelle (URL) von yahoo

ks	01. jun. 2016	rev 1.5
	komplette überarbeitung auf neue yahoo-api

ks	22. may. 2017	rev 1.6
	automatische wiederholung der Abfrage wenn keine Daten vorhanden sind
*)
END_FUNCTION_BLOCK
