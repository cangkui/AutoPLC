FUNCTION FILE_BLOCK_DEMO
VAR
    PT : NETWORK_BUFFER;
    fsd : FILE_SERVER_DATA;
    FILE_SERVER : FILE_SERVER;
    FILE_BLOCK : FILE_BLOCK;
    filename : STRING := 'config.ini';
    data : BYTE;
    error : BYTE;
    mode : BYTE;
    pos : UDINT;
END_VAR

END_FUNCTION
