import * as React from 'react';
import Button from '@mui/material/Button';
import SendIcon from '@mui/icons-material/Send';
import './App.css';
import Input from './components/Input';

function App() {
  return (
  <div>
    <Input value='409'/>
    <Input value='419'/>
    <Input value='429'/>
      <Button variant="contained" color="success" size="small" endIcon={<SendIcon />}>
        Send
      </Button>
  </div>
  )
}

export default App;
