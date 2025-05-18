FUNCTION HTTP_DEMO
VAR
    url_text : STRING := 'http://weather.yahooapis.com/forecastrss?p=94089&u=c';
    URL_DATA : URL; 
    IP_C : IP_C; 
    S_BUF : NETWORK_BUFFER; 
    R_BUF : NETWORK_BUFFER; 
    IP_CONTROL_1 : IP_CONTROL; 
    DNS_CLIENT : DNS_CLIENT; 
    HTTP_GET : HTTP_GET; 
    http_release : BOOL; 
    enable : BOOL := TRUE; 
    http_status : STRING; 
    header_start : UINT; 
    header_stop : UINT; 
    body_start : UINT; 
    body_stop : UINT; 
    http_get_done : BOOL; 
    http_get_error : DWORD; 
    COUNT_DR : OSCAT_BASIC.COUNT_DR; 
    cycles : DWORD; 
    TON_1 : TON; 
    F_TRIG : F_TRIG; 
END_VAR

END_FUNCTION
