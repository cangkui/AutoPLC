FUNCTION MB_SERVER_DEMO
VAR
    IP_CONTROL2_1 : IP_CONTROL2;    
    MB_SERVER_1 : MB_SERVER;        
    IP_C : IP_C;                    
    S_BUF : NETWORK_BUFFER_SHORT;    
    R_BUF : NETWORK_BUFFER_SHORT;    
    DATA : ARRAY[0..255] OF WORD;    
    VMAP : ARRAY[1..10] OF VMAP_DATA; 
    error : DWORD;                  
END_VAR

END_FUNCTION
