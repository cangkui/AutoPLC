FUNCTION DLOG_FILE_CSV_SMTP_DEMO
VAR
    cur_dt : DT;   
    enable : BOOL := TRUE;   
    trig_t : TIME := TIME#5s0ms;  
    trig_m : BOOL;   
    error_c : DWORD;   
    error_t : BYTE;   
    auto_close : TIME := TIME#10s0ms;  
    x1 : DLOG_DT;   
    x2 : DLOG_REAL;   
    x3 : DLOG_DINT;   
    x4 : DLOG_BOOL;   
    x5 : DLOG_STRING;   
    x : DLOG_DATA;   
    RTC_2 : oscat_basic.RTC_2;   
    DLOG_STORE_FILE_CSV : DLOG_STORE_FILE_CSV;   
    GEN_SIN_1 : oscat_basic.GEN_SIN;   
    COUNT_DR : oscat_basic.COUNT_DR;   
    DLOG_FILE_TO_SMTP : DLOG_FILE_TO_SMTP;   
    smtp_timeout : TIME := TIME#30s0ms;  
    smtp_error_t : BYTE;   
    smtp_error_c : DWORD;   
    smtp_busy : BOOL;   
    smtp_done : BOOL;   
    smtp_dti : DT;   
    smtp_dti_offset : INT := 60;  
    smtp_server : STRING := '110876022:oscat123456@mail.gmx.net';  
    smtp_mailfrom : STRING := 'oscat@gmx.net;Station_01';   
    smtp_mailto : STRING[OSCAT_BASIC.STRING_LENGTH] := 'oscat@gmx.net';   
    smtp_subject : STRING := 'Das ist der Betreff';   
    smtp_body : STRING[OSCAT_BASIC.STRING_LENGTH] := 'Das ist der Email-Text';   
    smtp_delete : BOOL := TRUE;  
    smtp_retry : INT := 0;  
    smtp_retry_time : TIME := TIME#1m0s0ms;  
    save_data : DLOG_SAVE; 
END_VAR

END_FUNCTION
