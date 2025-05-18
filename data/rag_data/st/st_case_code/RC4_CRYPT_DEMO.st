FUNCTION RC4_CRYPT_DEMO
VAR
    RC4_CRYPT_STREAM : RC4_CRYPT_STREAM; 
    buf : ARRAY[0..63] OF BYTE := [110, 60, 3, 219]; // crypt  
    mode : INT; 
    key : STRING := '1234'; 
    size : UDINT := 4; 
    pos : UDINT; 
END_VAR

END_FUNCTION
