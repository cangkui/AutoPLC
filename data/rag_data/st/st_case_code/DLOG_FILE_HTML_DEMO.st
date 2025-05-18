FUNCTION DLOG_FILE_HTML_DEMO
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
    DLOG_STORE_FILE_HTML : DLOG_STORE_FILE_HTML;  
    GEN_SIN_1 : oscat_basic.GEN_SIN;  
    COUNT_DR : oscat_basic.COUNT_DR;  
    html_caption : STRING := 'Das ist der Titel';  
    html_tr_even : STRING := 'BGCOLOR=#B3B7FF';  
    html_tr_odd : STRING := 'BGCOLOR=#E0E0E0';  
    html_tr_head : STRING := 'BGCOLOR=#FFFF40';  
    html_table : STRING := 'BORDER="1"';  
END_VAR

VAR RETAIN
    save_data : DLOG_SAVE;  
END_VAR

END_FUNCTION
