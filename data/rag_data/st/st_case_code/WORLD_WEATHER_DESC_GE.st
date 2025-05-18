FUNCTION_BLOCK WORLD_WEATHER_DESC_GE
VAR_IN_OUT
    WW : WORLD_WEATHER_DATA; // World Weather Data input/output variable
END_VAR

VAR_INPUT
    ACTIVATE : BOOL; // Activate control signal
END_VAR

VAR
    act_last : BOOL; // Last activation state
    x : INT;         // X coordinate or value
    y : INT;         // Y coordinate or value
    str : STRING(60); // String variable with a length of 60 characters
    i : INT;         // Iterator variable
END_VAR
IF ACTIVATE AND act_last = FALSE THEN
	i := 0;
	FOR y := 1 TO 12 DO
		CASE y OF
		1,3,5,7,9:		x :=  WW.DAY[i].WEATHER_CODE;
		2,4,6,8,10:	WW.DAY[i].WEATHER_DESC := str;
					i := i + 1;
		11:			x :=  WW.CUR.WEATHER_CODE;
		12:			WW.CUR.WEATHER_DESC := str;
		END_CASE;

		CASE x OF
			113	:	str := 'heiter/sonnig';
			116	:	str := 'stellenweise wolkig';
			119	:	str := 'wolkig';
			122	:	str := 'bewölkt';
			143	:	str := 'Nebel';
			176	:	str := 'stellenweise nahegelegener Regen';
			179	:	str := 'stellenweise nahegelegener Schneefall';
			182	:	str := 'stellenweise nahegelegener Eisregen';
			185	:	str := 'stellenweise gefrierender Niesel';
			200	:	str := 'nahegelegene Sturm';
			227	:	str := 'Schneetreiben';
			230	:	str := 'Schneesturm';
			248	:	str := 'Nebel';
			260	:	str := 'gefrierender Nebel';
			263	:	str := 'stellenweise leichter Nieselregen';
			266	:	str := 'leichter Niesel';
			281	:	str := 'gefrierender Nieselregen';
			284	:	str := 'stark gefrierender Nieselregen';
			293	:	str := 'stellenweise leichter Regen';
			296	:	str := 'leichter Regen';
			299	:	str := 'zeitweise etwas Regen';
			302	:	str := 'mäßiger Regen';
			305	:	str := 'zeitweise etwas starker Regen';
			308	:	str := 'starker Regen';
			311	:	str := 'leichter gefrierender Regen';
			314	:	str := 'mäßiger oder starker Eisregen';
			317	:	str := 'leichter Eisregen';
			320	:	str := 'mäßiger oder starker Eisregen';
			323	:	str := 'stellenweise leichter Schneefall';
			326	:	str := 'leichter Scheefall';
			329	:	str := 'stellenweise moderater Schneefall';
			332	:	str := 'mäßiger Schnee';
			335	:	str := 'stellenweise starker Scheefall';
			338	:	str := 'starker Schneefall';
			350	:	str := 'Hagel';
			353	:	str := 'leichte Regenfälle';
			356	:	str := 'mäßiger oder starker Regenfall';
			359	:	str := 'sintflutartige Regenfälle';
			362	:	str := 'leichter Eisregen';
			365	:	str := 'mäßiger oder starker Eisregen';
			368	:	str := 'leichte Schneeschauer';
			371	:	str := 'mäßiger oder starker Schneefall';
			374	:	str := 'leichter Hagel';
			377	:	str := 'mäßiger oder starker Hagel';
			386	:	str := 'stellenweise leichter Regen in Gewittergebiet';
			389	:	str := 'mäßiger oder starker Regen in Gewittergebiet';
			392	:	str := 'stellenweise leichter Schneefall in Gewittergebiet';
			395	:	str := 'mäßiger oder starker Schnee in Gewittergebiet';
		ELSE
			str := 	'nicht verfügbar';
		END_CASE;
	END_FOR;
END_IF;
act_last := ACTIVATE;

(* revision history
ks	04. may. 2011	rev 1.0
	original version
*)
END_FUNCTION_BLOCK
