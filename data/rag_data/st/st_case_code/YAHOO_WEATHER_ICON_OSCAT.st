FUNCTION_BLOCK YAHOO_WEATHER_ICON_OSCAT
VAR_IN_OUT
    YW : YAHOO_WEATHER_DATA;    
END_VAR

VAR_INPUT
    ACTIVATE : BOOL;    
END_VAR

VAR
    dat : ARRAY[0..48] OF INT := [4, 2, 2, 4, 17, 1, 3, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 5, 3, 6, 7, 14, 8, 9, 9, 10, 9, 11, 12, 11, 12, 13, 14, 15, 16, 4, 14, 4, 17, 2, 2, 3, 3, 3, 12, 2, 3, 4, 0];
    act_last : BOOL;    
    x : INT;    
    idx : INT;    
END_VAR
IF ACTIVATE AND act_last = FALSE THEN
	x := MIN(YW.CUR_CONDITIONS_CODE,48);
	YW.CUR_CONDITIONS_ICON := dat[x];
	x := MIN(YW.FORECAST_TODAY_CODE,48);
	YW.FORECAST_TODAY_ICON := dat[x];
	FOR idx := 1 TO 9 DO
		x := MIN(YW.FORECAST_DAY[idx].CODE,48);
		YW.FORECAST_DAY[idx].ICON := dat[x];
	END_FOR;
END_IF;
act_last := ACTIVATE;

(* revision history
ks	04. may. 2011	rev 1.0
	original version

ks	01. jun. 2016	rev 1.1
	komplette überarbeitung auf neue yahoo-api
*)
END_FUNCTION_BLOCK
