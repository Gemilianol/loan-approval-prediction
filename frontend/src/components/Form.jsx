'use client';
import React, { useState } from 'react';
import { Form } from '@base-ui-components/react/form';
import { Field } from '@base-ui-components/react/field';
import Button from '@mui/material/Button';
import SendIcon from '@mui/icons-material/Send';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import styles from './index.module.css';
import Typography from '@mui/material/Typography';

function DefaultForm() {
  
  // Data Fields (Min data values as default Integers)
  const [data, setData] = useState({
    'income': 30050, 
    'credit_score': 300, 
    'loan_amount': 1020, 
    'years_employed': 0, 
    'points': 10
  })

  // Handling changes on the data
  function handleChange(event){
    const {name, value} = event.target;

  // Use parseFloat for potentially decimal numbers, or parseInt for integers
  // Note: The second argument (10) specifies base 10 (decimal)
  let processedValue = parseInt(value);

  // A common safeguard: If the input is empty ('') or results in NaN, 
  // you might want to send null or 0, depending on your model's requirement.
  if (isNaN(processedValue)) {
      processedValue = 0; // or null, if your model handles nulls
  }

  setData(prevState => ({
      ...prevState,
      [name]: processedValue
    })); 
  };

  // Handle Submit async
  async function handleSubmit(event){
    // Prevent default bahavior of the browser (reloded)
    event.preventDefault();

    // Disabled the button for now:
    setLoading(true);

    try {
      const response = await fetch('http://localhost:2000/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
      });

      // If I get an error then:
      if (!response.ok) {
        // Key-Value pairs
        const errorDetails = await response.json();

        // Update the state
        setErrors(errorDetails);

        console.log(errors);

        // Render the Card component:
        setResult(true); 

        // Enable the button again:
        setLoading(false);
      } else {
        const prediction = await response.json();

        // Get the prediction to render it:
        setResponse(prediction);

        console.log(prediction);

        // Render the Card component:
        setResult(true); 

        // Clean the Form for the next request:
        setData({
              'income': 30050, 
              'credit_score': 300, 
              'loan_amount': 1020, 
              'years_employed': 0, 
              'points': 10
        });
        
        // Enable the button again:
        setLoading(false);
      } 
      } catch(error) {
        console.error("Something happened through the submit: ", error);
    }
  };

  // // Backend's response
  const [response, setResponse] = useState({});

  // // Backend's Errors:
  const [errors, setErrors] = useState({});

  // Conditional rendering for Button
  const [loading, setLoading] = useState(false);

  // Conditional rendering for Card
  const [result, setResult] = useState(false);

  return (
    <Form 
    className={styles.Form} 
    errors={errors} 
    onSubmit={handleSubmit}
    >

      <Field.Root name="income" className={styles.Field}>
        <Field.Label className={styles.Label}>Anual Income (USD)</Field.Label>
        <Field.Control
          type="number"
          required
          // If you want to force always get values the backend:
          defaultValue={data.income}
          placeholder={data.income}
          className={styles.Input} 
          onChange={handleChange}
        />
        <Field.Error className={styles.Error}/>
        <Field.Description className={styles.Description}>Your Current Anual Income in USD</Field.Description>
      </Field.Root>

      <Field.Root name="credit_score" className={styles.Field}>
        <Field.Label className={styles.Label}>Credit Score</Field.Label>
        <Field.Control
          type="number"
          required
          defaultValue= {data.credit_score}
          placeholder={data.credit_score}
          className={styles.Input}
          onChange={handleChange}
        />
        <Field.Error className={styles.Error}/>
        <Field.Description className={styles.Description}>Your Current Credit Score</Field.Description>
      </Field.Root>

      <Field.Root name="loan_amount" className={styles.Field}>
        <Field.Label className={styles.Label}>Loan Amount (USD)</Field.Label>
        <Field.Control
          type="number"
          required
          defaultValue={data.loan_amount}
          placeholder={data.loan_amount}
          className={styles.Input}
          onChange={handleChange}
        />
        <Field.Error className={styles.Error}/>
        <Field.Description className={styles.Description}>The Loan Amount in USD</Field.Description>
      </Field.Root>

      <Field.Root name="years_employed" className={styles.Field}>
        <Field.Label className={styles.Label}>Years Employed</Field.Label>
        <Field.Control
          type="number"
          required
          defaultValue={data.years_employed}
          placeholder={data.years_employed}
          className={styles.Input}
          onChange={handleChange}
        />
        <Field.Error className={styles.Error}/>
        <Field.Description className={styles.Description}>Years Employed in Your Actual Job</Field.Description>
      </Field.Root>

      <Field.Root name="points" className={styles.Field}>
        <Field.Label className={styles.Label}>Points</Field.Label>
        <Field.Control
          type="number"
          required
          defaultValue={data.points}
          placeholder={data.points}
          className={styles.Input}
          onChange={handleChange}
        />
        <Field.Error className={styles.Error}/>
        <Field.Description className={styles.Description}>Points</Field.Description>
      </Field.Root>

      <Button type="submit" 
      disabled={loading} 
      autoFocus
      className={styles.Button} 
      variant="contained" 
      endIcon={<SendIcon />}
      >
        <Typography 
        align="left"
        noWrap= "true"
        variant='caption'>
          {loading ? 'Submitting' : 'Simulate Credit Approval'}
        </Typography>
        
      </Button>

      {/* Only render a Card component If you've already have a prediction OR an error  */}

      {result && response.Result ? <Card className={styles.Card}>
        <CardContent className={styles.CardContent}>{response.Result}</CardContent></Card> : null}

      {result && errors.value ? <Card className={styles.Card}><CardContent className={styles.CardContent}>{errors.value}</CardContent></Card> : null}

    </Form>
  );
}

export default DefaultForm;