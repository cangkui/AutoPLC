FUNCTION STRING_TO_URL : URL
VAR_INPUT
    STR: STRING(OSCAT_BASIC.STRING_LENGTH);
    DEFAULT_PROTOCOL: STRING(10);
    DEFAULT_PATH: STRING(80);
END_VAR

VAR
    pos: INT;
    pos2: INT;
    x: STRING(OSCAT_BASIC.STRING_LENGTH);
END_VAR
(*

http://hans:geheim@www.example.org:80/demo/example.cgi?land=de&stadt=aa#abschnitt1
|       |      |         |         |   |                   |               |
|       |      |         Host      |   Pfad                Query           Anker
|       |      Passwort            Port
|       Benutzer
Protokoll

*)

(* check if string is empty and return empty url *)
IF STR = '' THEN RETURN; END_IF;

(* extract the protocol *)
(* first we check if the optional protocol is specified ending with a :// *)
pos := FIND(STR, '://');
IF pos > 0 THEN
	(* store protocol and restore the rest of url in temp *)
	STRING_TO_URL.PROTOCOL := LEFT(STR, pos - 1);
	x := RIGHT(STR, LEN(STR) - pos - 2);
ELSE
	x := STR;
	STRING_TO_URL.PROTOCOL := DEFAULT_PROTOCOL;
END_IF;

(* extract user and password if present ending with a @ *)
pos := OSCAT_BASIC.FINDB(STR1:=x,STR2:='@'); (* Find last @ *)

IF pos > 0 THEN
	(* extract user *)
	pos2 := FIND(x,':')+1;
	IF pos2 > 1 AND pos2 <= pos THEN
		STRING_TO_URL.USER := LEFT(x,pos2 - 2);
		STRING_TO_URL.PASSWORD := MID(x, pos - pos2, pos2);
	ELSE
		STRING_TO_URL.USER := LEFT(x, pos - 2);
		STRING_TO_URL.PASSWORD := '';
	END_IF;
	x := RIGHT(x, LEN(x) - pos);
ELSE
	STRING_TO_URL.USER := '';
	STRING_TO_URL.PASSWORD := '';
END_IF;

(* extract anchor *)
pos := FIND(x, '#');
IF pos > 0 THEN
	STRING_TO_URL.ANCHOR := RIGHT(x, LEN(x) - pos);
	x := LEFT(x, pos - 1);
ELSE
	STRING_TO_URL.ANCHOR := '';
END_IF;

(* extract query  *)
pos := FIND(x, '?');
IF pos > 0 THEN
	STRING_TO_URL.QUERY := RIGHT(x, LEN(x) - pos);
	x := LEFT(x, pos - 1);
ELSE
	STRING_TO_URL.QUERY := '';
END_IF;

(* extract path  *)
pos := FIND(x, '/');
IF pos > 0 THEN
	STRING_TO_URL.PATH := RIGHT(x, LEN(x) - pos + 1);
	x := LEFT(x, pos - 1);
ELSE
	STRING_TO_URL.PATH := DEFAULT_PATH;
END_IF;

(* extract port and leave domain *)
pos := FIND(x, ':');
IF pos > 0 THEN
	STRING_TO_URL.PORT := STRING_TO_WORD(RIGHT(x, LEN(x) - pos));
	STRING_TO_URL.DOMAIN := LEFT(x, pos - 1);
ELSE
	STRING_TO_URL.PORT := 0;
	STRING_TO_URL.DOMAIN := x;
END_IF;

(* revision history
hm		18. apr 2008	rev 1.0
	original version

ks		20. dec. 2010	rev 1.1
	stringlength 250 , var x

ks		27. oct. 2012	rev 1.2
	var_temp zu var geändert

ks		04. mar. 2016	rev 1.3
	@ zeichen wird nun von rechts nach links gesucht
	ermöglicht username und password mit @ zeichen
*)
END_FUNCTION
