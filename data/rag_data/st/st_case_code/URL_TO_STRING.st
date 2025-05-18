FUNCTION URL_TO_STRING : string
VAR_INPUT
    IN : URL;
END_VAR
(*

http://hans:geheim@www.example.org:80/demo/example.cgi?land=de&stadt=aa#abschnitt1
|         |         |                  |                                |   |                                    |                                    |
|         |         |                 Host                         |   Pfad                            Query                           Anker
|         |         Passwort                                  Port
|        Benutzer
Protokoll

*)

(* attach protocol *)
IF IN.PROTOCOL <> '' THEN URL_TO_STRING := CONCAT(IN.PROTOCOL,'://'); END_IF;

(* attach user and password *)
IF IN.USER <> '' THEN
	URL_TO_STRING := CONCAT(URL_TO_STRING, IN.USER);
	IF IN.PASSWORD <> '' THEN
		URL_TO_STRING := CONCAT(URL_TO_STRING, ':');
		URL_TO_STRING := CONCAT(URL_TO_STRING, IN.PASSWORD);
	END_IF;
	URL_TO_STRING := CONCAT(URL_TO_STRING, '@');
END_IF;

(* attach domain *)
URL_TO_STRING := CONCAT(URL_TO_STRING, IN.DOMAIN);

(* attach Port *)
IF IN.PORT > 0 THEN
	URL_TO_STRING := CONCAT(URL_TO_STRING, ':');
	URL_TO_STRING := CONCAT(URL_TO_STRING, WORD_TO_STRING(IN.PORT));
END_IF;

(* attach path *)
URL_TO_STRING := CONCAT(URL_TO_STRING, IN.PATH);

(* attach query *)
IF IN.QUERY <> '' THEN
	URL_TO_STRING := CONCAT(URL_TO_STRING, '?');
	URL_TO_STRING := CONCAT(URL_TO_STRING, IN.QUERY);
END_IF;

(* attach anchor *)
IF IN.ANCHOR <> '' THEN
	URL_TO_STRING := CONCAT(URL_TO_STRING, '#');
	URL_TO_STRING := CONCAT(URL_TO_STRING, IN.ANCHOR);
END_IF;

(* revision history
hm	18. apr 2008	rev 1.0
	original version

ks	10. mai. 2008	rev 1.1
	stringlength 250
*)
END_FUNCTION
