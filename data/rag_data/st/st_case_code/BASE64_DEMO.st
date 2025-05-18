FUNCTION BASE64_DEMO
VAR
    text1 : STRING(144) := '<4036.1076324381@guildenstern.zeitform.de>';
    text2 : STRING(192);
    text3 : STRING(144);
    BASE64_DECODE_STR : BASE64_DECODE_STR; (* Assuming BASE64_DECODE_STR is a previously defined type *)
    BASE64_ENCODE_STR : BASE64_ENCODE_STR; (* Assuming BASE64_ENCODE_STR is a previously defined type *)
    start : BOOL := TRUE;
    done1 : BOOL;
    done2 : BOOL;
END_VAR

END_FUNCTION
