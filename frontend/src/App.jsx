import * as React from 'react';
import Button from '@mui/material/Button';
import SendIcon from '@mui/icons-material/Send';
import './App.css';

function App() {
  return (
  <div>
      <Button variant="contained" color="success" size="small" endIcon={<SendIcon />}>
        Send
      </Button>
  </div>
  )
}

export default App;
