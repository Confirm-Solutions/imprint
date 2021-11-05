import Canvas from './canvas';
import { getTestMatrices, MatrixData, PlotType } from './data';
import React from 'react';
import './App.css';
import { FormControl, FormLabel, FormControlLabel, Radio, RadioGroup, Checkbox } from '@mui/material';

const numLayers = 6;
const layerNames = ["Monte Carlo Type I error estimates", "0th order upper bound", "Max gradient estimates", "1st order upper bound", "2nd order upper bound", "Total upper bound"]

function App() {
  const [plotType, setPlotType] = React.useState<PlotType>("surface");
  const [data, setData] = React.useState<MatrixData>();

  const getDefaultState = () => {
    let s: boolean[] = new Array(numLayers).fill(false)
    s[numLayers - 1] = true;
    return s
  }
  const [checkboxStates, setCheckboxStates] = React.useState<boolean[]>(getDefaultState())
  const fetchData = React.useCallback(
    async function () {
      setData(await getTestMatrices())
    }, []
  )

  React.useEffect(() => {
    fetchData()
  }, [plotType, checkboxStates, fetchData])

  const checkboxes = checkboxStates.map((value, i) => {
    return <FormControlLabel key={i} label={layerNames[i]} control={<Checkbox checked={value} onChange={(_, v) => {
      let newState = Array.from(checkboxStates);
      newState[i] = v;
      setCheckboxStates(newState)
    }} />} />
  })


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
      <div className="left">
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
        <FormControl>
          <FormLabel id="demo-radio-buttons-group-label">Layers Shown</FormLabel>
          {checkboxes}
        </FormControl>
      </div>
      <div className="right">
        <article>
          <header>
            <h1>Simulated Family-Wise Error Rates</h1>
          </header>
          <main>
            {data ?
              <Canvas
                data={data}
                plotType={plotType}
                checkboxStates={checkboxStates}
                layerNames={layerNames}
              ></Canvas>
              : null}
          </main>
          <footer>
          </footer>
        </article>
      </div>
    </div>
  );
}

export default App;
