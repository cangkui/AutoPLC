FUNCTION MD5_CRAM_AUTH_DEMO
VAR
    run : BOOL := TRUE;
    timestamp : STRING[64] := 'PDQwMzYuMTA3NjMyNDM4MUBndWlsZGVuc3Rlcm4uemVpdGZvcm0uZGU+';
    password : STRING := 'geheimes_passwort';
    username : STRING := 'benutzer@zeitform.de';
    auth_key : STRING[OSCAT_BASIC.STRING_LENGTH];
    MD5_CRAM_AUTH : MD5_CRAM_AUTH; // Assuming MD5_CRAM_AUTH is a defined derived type
END_VAR

END_FUNCTION
