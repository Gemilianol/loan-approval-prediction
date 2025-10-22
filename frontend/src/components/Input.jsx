import React from "react";
import TextField from '@mui/material/TextField';

function Input(props) {
    return(
    <box sx={{ m: 3, width: '25ch' }}>
        <TextField 
        required error id="standard-basic" 
        label="Standard: Mean Value" 
        variant="standard"
        value={props.value}
        defaultValue={'250'}
        helperText="The value must be integer"/>
    </box>
    )
};

export default Input;