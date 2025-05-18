FUNCTION DNS_DYN_DEMO
VAR
    DNS_DYN : DNS_DYN;
    IP_C : IP_C;
    S_BUF : NETWORK_BUFFER;
    R_BUF : NETWORK_BUFFER;
    IP_CONTROL : IP_CONTROL;
    enable : BOOL;
    done : BOOL;
    error_c : DWORD;
    error_t : BYTE;
    busy : BOOL;
    COUNT_DR : oscab_basic.COUNT_DR;
    z : DWORD;
    GET_WAN_IP : GET_WAN_IP;
    t1 : TON;
    done2 : BOOL;
    error_c2 : DWORD;
    error_t2 : BYTE;
    update : BOOL;
    new_ip : BOOL;
    wan_ip4 : DWORD;
END_VAR

END_FUNCTION
