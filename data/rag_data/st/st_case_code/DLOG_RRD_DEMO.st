FUNCTION DLOG_RRD_DEMO
VAR
    cur_dt : DT;
    trig_t : TIME := TIME#50s0ms;
    trig_m : BOOL;
    rrd_error_c : DWORD;
    rrd_error_t : BYTE;
    rrd_enable : BOOL := TRUE;
    rrd_url : STRING(OSCAT_BASIC.STRING_LENGTH) := '192.168.178.21/rrd/test_rrd2.php?rrd_db=test.rrd&value=';
    rrd_timeout : TIME := TIME#5s0ms;
    RTC_2 : oscat_basic.RTC_2;
    DLOG_STORE_RRD : DLOG_STORE_RRD;
    x : DLOG_DATA;
    x1 : DLOG_DINT;
    x2 : DLOG_DINT;
    x3 : DLOG_REAL;
    x4 : DLOG_REAL;
    x5 : DLOG_REAL;
    x6 : DLOG_REAL;
    x7 : DLOG_REAL;
    GEN_SIN_1 : oscat_basic.GEN_SIN;
    GEN_SIN_2 : oscat_basic.GEN_SIN;
    GEN_SIN_3 : oscat_basic.GEN_SIN;
    GEN_RMP_1 : oscat_basic.GEN_RMP;
    GEN_RMP_2 : oscat_basic.GEN_RMP;
END_VAR

END_FUNCTION
