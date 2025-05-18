FUNCTION IRTRANS_DEMO
VAR
    IP_C : IP_C;
    S_BUF : NETWORK_BUFFER_SHORT;
    R_BUF : NETWORK_BUFFER_SHORT;
    IP_CONTROL2 : IP_CONTROL2;
    IRTRANS_DECODE : IRTRANS_DECODE;
    IRTRANS_RCV_4 : IRTRANS_RCV_4;
    DRIVER_4 : oscat_basic.DRIVER_4;
    IRTRANS_SERVER : IRTRANS_SERVER;
    IRTRANS_SND_8_1 : IRTRANS_SND_8;
    IRTRANS_SND_8_2 : IRTRANS_SND_8;
    Q0 : BOOL;
    Q1 : BOOL;
    Q2 : BOOL;
    Q3 : BOOL;
    error : BOOL;
    error_server : DWORD;
    Send_1_0 : BOOL := TRUE;
    Send_1_1 : BOOL;
    Send_1_2 : BOOL;
    Send_1_3 : BOOL;
    Send_1_4 : BOOL;
    Send_1_5 : BOOL;
    Send_1_6 : BOOL;
    Send_1_7 : BOOL;
    Send_2_0 : BOOL := TRUE;
    Send_2_1 : BOOL;
    Send_2_2 : BOOL;
    Send_2_3 : BOOL;
    Send_2_4 : BOOL;
    Send_2_5 : BOOL;
    Send_2_6 : BOOL;
    Send_2_7 : BOOL;
    R_ENABLE : BOOL;
    S_ENABLE : BOOL;
    MODE : BOOL := FALSE;
    KEY_1 : BYTE;
    KEY_2 : BYTE;
END_VAR

END_FUNCTION
