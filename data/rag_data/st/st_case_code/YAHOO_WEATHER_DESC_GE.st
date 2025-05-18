FUNCTION_BLOCK YAHOO_WEATHER_DESC_GE
VAR_IN_OUT
    YW : YAHOO_WEATHER_DATA;
END_VAR

VAR_INPUT
    ACTIVATE : BOOL;
END_VAR

VAR
    act_last : BOOL;
    x : INT;
    y : INT;
    str : STRING[40];
END_VAR
IF ACTIVATE AND act_last = FALSE THEN
	FOR y := 1 TO 12 DO
		CASE y OF
		01:	x := YW.CUR_CONDITIONS_CODE;

		02:	YW.CUR_CONDITIONS_TEXT := str;
			x := YW.FORECAST_TODAY_CODE;
		03:	YW.FORECAST_TODAY_TEXT := str;
			x := YW.FORECAST_DAY[1].CODE;
		04:	YW.FORECAST_DAY[1].TEXT := str;
			x := YW.FORECAST_DAY[2].CODE;
		05:	YW.FORECAST_DAY[2].TEXT := str;
			x := YW.FORECAST_DAY[3].CODE;
		06:	YW.FORECAST_DAY[3].TEXT := str;
			x := YW.FORECAST_DAY[4].CODE;
		07:	YW.FORECAST_DAY[4].TEXT := str;
			x := YW.FORECAST_DAY[5].CODE;
		08:	YW.FORECAST_DAY[5].TEXT := str;
			x := YW.FORECAST_DAY[6].CODE;
		09:	YW.FORECAST_DAY[6].TEXT := str;
			x := YW.FORECAST_DAY[7].CODE;
		10:	YW.FORECAST_DAY[7].TEXT := str;
			x := YW.FORECAST_DAY[8].CODE;
		11:	YW.FORECAST_DAY[8].TEXT := str;
			x := YW.FORECAST_DAY[9].CODE;
		12:	YW.FORECAST_DAY[9].TEXT := str;
		END_CASE;

		CASE x OF
		00:	str := 'Wirbelsturm';
		01:	str := 'Tropensturm';
		02:	str := 'Wirbelsturm';
		03:	str := 'starkes Gewitter ';
		04:	str := 'Gewitter';
		05:	str := 'Schneeregen';
		06:	str := 'Schneeregen';
		07:	str := 'Schneeregen';
		08:	str := 'gefrierender Nieselregen';
		09:	str := 'Nieselregen';
		10:	str := 'gefrierender Regen';
		11:	str := 'Regenfälle';
		12:	str := 'Regenfälle';
		13:	str := 'Schneegestöber';
		14:	str := 'leichte Schneefälle';
		15:	str := 'Schneetreiben';
		16:	str := 'Schnee';
		17:	str := 'Hagel';
		18:	str := 'Schneeregen';
		19:	str := 'Staub';
		20:	str := 'nebelig';
		21:	str := 'Dunst';
		22:	str := 'rauchig';
		23:	str := 'stürmisch';
		24:	str := 'windig';
		25:	str := 'kalt';
		26:	str := 'bewölkt';
		27:	str := 'meist bewölkt';
		28:	str := 'meist bewölkt';
		29:	str := 'teilweise bewölkt';
		30:	str := 'teilweise bewölkt';
		31:	str := 'klar';
		32:	str := 'sonnig';
		33:	str := 'heiter';
		34:	str := 'heiter';
		35:	str := 'Regen und Hagel';
		36:	str := 'heiß';
		37:	str := 'örtliche Gewitter';
		38:	str := 'vereinzelte Gewitter';
		39:	str := 'vereinzelte Gewitter';
		40:	str := 'vereinzelte Regenfälle';
		41:	str := 'schwere Schneefälle';
		42:	str := 'vereinzelte Schneefälle';
		43:	str := 'schwere Schneefälle';
		44:	str := 'teilweise bewölkt';
		45:	str := 'gewittrige Regenfälle';
		46:	str := 'Schneefälle';
		47:	str := 'örtliche gewittrige Regenfälle';
		ELSE
			str := 'nicht verfügbar';
		END_CASE;
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
