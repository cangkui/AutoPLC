FUNCTION SMTP_CLIENT_DEMO
VAR
    SMTP_CLIENT : STRING := 'oscat_test%freenet.de:oscat123456@mx.freenet.de';
    smtp_mailfrom : STRING := 'oscat_test@freenet.de';
    smtp_activate : BOOL;
    smtp_timeout : TIME := TIME#30s0ms;
    smtp_error_t : BYTE;
    smtp_error_c : DWORD;
    smtp_busy : BOOL;
    smtp_done : BOOL;
    smtp_dti : DT;
    smtp_dti_offset : INT := 60;
    smtp_files : STRING[OSCAT_BASIC.STRING_LENGTH] := '''';
    smtp_server : STRING := '110876022:oscat123456@mail.gmx.net';
    smtp_mailfrom : STRING := 'oscat@gmx.net;Station_01';
    smtp_mailto : STRING[OSCAT_BASIC.STRING_LENGTH] := 'oscat@gmx.net';
    smtp_subject : STRING := 'Das ist der Betreff';
    smtp_body : STRING[OSCAT_BASIC.STRING_LENGTH] := 'Das ist der Email-Text';
END_VAR

END_FUNCTION
