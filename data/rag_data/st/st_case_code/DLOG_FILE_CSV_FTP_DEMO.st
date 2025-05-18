FUNCTION DLOG_FILE_CSV_FTP_DEMO
VAR
    cur_dt: DT;
    enable: BOOL := TRUE;
    trig_t: TIME := TIME#5s0ms;
    trig_m: BOOL;
    error_c: DWORD;
    error_t: BYTE;
    auto_close: TIME := TIME#10s0ms;
    x1: DLOG_DT;
    x2: DLOG_REAL;
    x3: DLOG_DINT;
    x4: DLOG_BOOL;
    x5: DLOG_STRING;
    x: DLOG_DATA;
    RTC_2: oscat_basic.RTC_2;
    DLOG_STORE_FILE_CSV: DLOG_STORE_FILE_CSV;
    GEN_SIN_1: oscat_basic.GEN_SIN;
    COUNT_DR: oscat_basic.COUNT_DR;
    DLOG_FILE_TO_FTP: DLOG_FILE_TO_FTP;
    ftp_done: BOOL;
    ftp_busy: BOOL;
    ftp_error_c: DWORD;
    ftp_error_t: BYTE;
    ftp_url: STRING := 'ftp://581920:123456@oscat.atwebpages.com:2000/subordner/';
    ftp_activ: BOOL := FALSE;
    ftp_delete: BOOL := TRUE;
    ftp_retry: INT := 0;
    ftp_retry_time: TIME := TIME#1m0s0ms;
    ftp_timeout: TIME := TIME#30s0ms;
    save_data: DLOG_SAVE; // Retained variable
END_VAR

END_FUNCTION
