FUNCTION DLOG_FILE_XML_DEMO
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
    DLOG_STORE_FILE_XML: DLOG_STORE_FILE_XML;
    GEN_SIN_1: oscat_basic.GEN_SIN;
    COUNT_DR: oscat_basic.COUNT_DR;
    save_data: DLOG_SAVE; // Retain variable
END_VAR

END_FUNCTION
