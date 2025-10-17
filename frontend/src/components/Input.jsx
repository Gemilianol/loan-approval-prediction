import React from "react";
import TextField from '@mui/material/TextField';

function Input() {
    return(
    <box>
        <TextField required error id="standard-basic" label="Standard" variant="standard" helperText="The value must be integer"/>
    </box>
    )
};

export default Input;