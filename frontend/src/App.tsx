import Canvas from './canvas';
import { PlotType } from './canvas';
import React from 'react';
import './App.css';
import { FormControl, FormLabel, FormControlLabel, Radio, RadioGroup } from '@mui/material';


function App() {
  const [plotType, setPlotType] = React.useState<PlotType>("surface");

  function handleChange(event: React.ChangeEvent<HTMLInputElement>, value: string) {
    switch (value) {
      case "surface":
        return setPlotType(value);
      case "scatter3d":
        return setPlotType(value);
    }
  }

  return (
    <div className="row">
      <div className="column left">
        <FormControl>
          <FormLabel id="demo-radio-buttons-group-label">Plot Type</FormLabel>
          <RadioGroup
            aria-labelledby="demo-radio-buttons-group-label"
            defaultValue="surface"
            name="radio-buttons-group"
            onChange={handleChange}
          >
            <FormControlLabel value="surface" control={<Radio />} label="Surface" />
            <FormControlLabel value="scatter3d" control={<Radio />} label="Scatter" />
          </RadioGroup>
        </FormControl>
      </div>
      <div className="column middle">
        <article>
          <header>
            <h1>Simulated Family-Wise Error Rates</h1>
          </header>
          <main>
            <Canvas plotType={plotType}></Canvas>
          </main>
          <footer>
          </footer>
        </article>
      </div>
      <div className='column right'></div>
    </div>
  );
}

export default App;
