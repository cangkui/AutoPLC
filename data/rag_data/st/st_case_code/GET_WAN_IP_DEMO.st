FUNCTION GET_WAN_IP_DEMO
VAR
    GET_WAN_IP : GET_WAN_IP;   (* Derived type *)
    IP_C : IP_C;               (* Derived type *)
    S_BUF : NETWORK_BUFFER;     (* Derived type *)
    R_BUF : NETWORK_BUFFER;     (* Derived type *)
    IP_CONTROL_1 : IP_CONTROL;  (* Derived type *)
    activate : BOOL;           (* Boolean variable *)
    done : BOOL;               (* Boolean variable *)
    error_c : DWORD;          (* Double Word variable *)
    error_t : BYTE;           (* Byte variable *)
    wan_ip : STRING;          (* String variable *)
    new_ip : BOOL;            (* Boolean variable *)
END_VAR

END_FUNCTION
