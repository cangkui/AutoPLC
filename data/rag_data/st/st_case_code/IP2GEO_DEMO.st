FUNCTION IP2GEO_DEMO
VAR
    IP_C : IP_C;
    S_BUF : NETWORK_BUFFER;
    R_BUF : NETWORK_BUFFER;
    IP_CONTROL_1 : IP_CONTROL;
    CLK_PRG : oscat_basic.CLK_PRG;
    IP2GEO_1 : IP2GEO;
    GEO : IP2GEO_DATA;
    activate : BOOL := TRUE;
    done : BOOL;
    busy : BOOL;
    error_c : DWORD;
    error_t : BYTE;
END_VAR

END_FUNCTION
