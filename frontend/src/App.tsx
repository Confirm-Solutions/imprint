import Canvas from './canvas';
import { getTestMatrices, MatrixData, PlotType, parseCSV } from './data';
import React from 'react';
import './App.css';
import { FormControl, FormLabel, FormControlLabel, Radio, RadioGroup, Checkbox, Button } from '@mui/material';

const numLayers = 6;
const layerNames = ["Monte Carlo Type I error estimates", "0th order upper bound", "Max gradient estimates", "1st order upper bound", "2nd order upper bound", "Total upper bound"]

function App() {
  const [plotType, setPlotType] = React.useState<PlotType>("surface");
  const [data, setData] = React.useState<MatrixData>();

  function updateDataCallback(field: "pmat" | "bmat") {
    return (e: React.ChangeEvent<HTMLInputElement>) => {
      if (!e.target.files || data === undefined) return
      e.target.files[0].text().then(text => {
        let newData = { ...data }
        newData[field] = parseCSV(text)
        setData(newData)

        // https://stackoverflow.com/questions/19643265/second-use-of-input-file-doesnt-trigger-onchange-anymore
        e.target.value = ''
      })
    }
  }

  const getDefaultState = () => {
    let s: boolean[] = new Array(numLayers).fill(false)
    s[numLayers - 1] = true;
    return s
  }
  const [checkboxStates, setCheckboxStates] = React.useState<boolean[]>(getDefaultState())
  const fetchTestData = React.useCallback(
    async function () {
      setData(await getTestMatrices())
    }, []
  )
  React.useEffect(() => {
    fetchTestData()
  }, [fetchTestData])

  const checkboxes = checkboxStates.map((value, i) => {
    return <FormControlLabel
      key={i}
      label={layerNames[i]}
      control={
        <Checkbox
          checked={value}
          onChange={(_, v) => {
            let newState = Array.from(checkboxStates);
            newState[i] = v;
            setCheckboxStates(newState)
          }} />
      } />
  })


  function handlePlotTypeChange(_: React.ChangeEvent<HTMLInputElement>, value: string) {
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
            onChange={handlePlotTypeChange}
          >
            <FormControlLabel value="surface" control={<Radio />} label="Surface" />
            <FormControlLabel value="scatter3d" control={<Radio />} label="Scatter" />
          </RadioGroup>
        </FormControl>
        <FormControl>
          <FormLabel id="demo-radio-buttons-group-label">Layers Shown</FormLabel>
          {checkboxes}
        </FormControl>
        <div className='buttonContainer'>
          <Button
            variant="contained"
            component="label"
          >
            Upload B Matrix
            <input
              type="file"
              accept=".csv"
              hidden
              onChange={updateDataCallback("bmat")}
            />
          </Button>
          <Button
            variant="contained"
            component="label"
          >
            Upload P Matrix
            <input
              type="file"
              accept=".csv"
              hidden
              onChange={updateDataCallback("pmat")}
            />
          </Button>
        </div>
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
