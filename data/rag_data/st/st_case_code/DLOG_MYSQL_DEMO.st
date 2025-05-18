FUNCTION DLOG_MYSQL_DEMO
VAR
    cur_dt : DT;
    enable : BOOL := TRUE;
    trig_t : TIME := TIME#5s0ms;
    trig_m : BOOL;
    error_c : DWORD;
    error_t : BYTE;
    x1 : DLOG_DT;
    x2 : DLOG_REAL;
    x3 : DLOG_DINT;
    x4 : DLOG_BOOL;
    x5 : DLOG_STRING;
    x : DLOG_DATA;
    mysql_info : MYSQL_INFO;
    RTC_2 : oscat_basic.RTC_2;
    GEN_SIN_1 : oscat_basic.GEN_SIN;
    COUNT_DR : oscat_basic.COUNT_DR;
    DLOG_STORE_MYSQL : DLOG_STORE_MYSQL;
    url : STRING[OSCAT_BASIC.STRING_LENGTH] := 'myuser:mypassword@192.168.178.21';
    db_name : STRING[64] := 'oscat_db';
    tb_name : STRING[64] := 'oscat_tb';
    auto_delete : UDINT := 0;
    timeout : TIME := TIME#10s0ms;
END_VAR

END_FUNCTION
