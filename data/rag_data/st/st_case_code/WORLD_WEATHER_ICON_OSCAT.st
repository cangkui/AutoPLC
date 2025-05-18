FUNCTION_BLOCK WORLD_WEATHER_ICON_OSCAT
VAR_IN_OUT
    WW : WORLD_WEATHER_DATA; 
END_VAR_IN_OUT

VAR_INPUT
    ACTIVATE : BOOL; 
END_VAR_INPUT

VAR
    dat : ARRAY[0..41] OF INT := [0, 14, 12, 9, 9, 0, 7, 7, 0, 2, 2, 3, 3, 1, 0, 0, 17, 4, 4, 3, 3, 1, 0, 0, 17, 2, 2, 3, 3, 1, 0, 0, 17, 2, 4, 3, 3, 1, 0, 0, 17, 0];
    act_last : BOOL; 
    x : INT; 
    i : INT; 
END_VAR
IF ACTIVATE AND act_last = FALSE THEN
	x := LIMIT(0,WW.CUR.WEATHER_ICON,41);
	WW.CUR.WEATHER_ICON := dat[x];
	FOR i := 0 TO 4 DO
		x := LIMIT(0,WW.DAY[i].WEATHER_ICON,41);
		WW.DAY[i].WEATHER_ICON := dat[x];
	END_FOR;
END_IF;
act_last := ACTIVATE;

(* revision history
ks	04. may. 2011	rev 1.0
	original version
*)
END_FUNCTION_BLOCK
