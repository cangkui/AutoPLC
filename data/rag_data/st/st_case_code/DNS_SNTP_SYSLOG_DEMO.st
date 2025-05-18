FUNCTION DNS_SNTP_SYSLOG_DEMO
VAR
    IP_C : IP_C;
    S_BUF : NETWORK_BUFFER;
    R_BUF : NETWORK_BUFFER;
    IP_CONTROL_1 : IP_CONTROL;
    DNS_CLIENT : DNS_CLIENT;
    SNTP_CLIENT : SNTP_CLIENT;
    SNTP_SERVER : SNTP_SERVER;
    COUNT_DR : oscat_basic.COUNT_DR;
    RTC_2 : oscat_basic.RTC_2;
    count_cycles : DWORD;
    dns_error : DWORD;
    sntp_error : DWORD;
    freigabe : BOOL;
    dso : BOOL;
    xms : INT;
    sntp_srv_enable : BOOL;
    cur_UDT : DT;
    SYS_LOG : SYS_LOG;
    sys_log_option : BYTE := 2#100;
    TON_1 : TON;
    tp_sys_log : BOOL;
    error_sys_log : DWORD;
    sys_time : TIME;
    cur_LOCAL_DT : DT;
END_VAR

END_FUNCTION
