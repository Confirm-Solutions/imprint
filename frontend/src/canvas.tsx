import React from 'react';
import Plotly from 'plotly.js-dist';
import { PlotType, MatrixData, getPlotlyData } from './data'


export interface CanvasState {
    plotType: PlotType
    checkboxStates: boolean[]
    data: MatrixData
    layerNames: string[]
    colorscale: string
}

async function plot(context: HTMLDivElement, height: number, width: number, state: CanvasState) {
    let method = Plotly.newPlot
    if ('data' in context) {
        console.log('updating plot in place')
        method = Plotly.react
    }

    var data = getPlotlyData(state.data, state.plotType, state.checkboxStates, state.layerNames)
    const layout: any = {
        height: height,
        coloraxis: { colorscale: state.colorscale },
        legend: {
            font: {
                family: '"Roboto","Helvetica","Arial",sans-serif',
                size: 18
            },
        }

    }
    await method(context, data, layout, { responsive: true });
}


function Canvas(props: CanvasState) {
    // https://stackoverflow.com/a/67906087
    const canvas = React.useCallback((node: HTMLDivElement) => {
        if (node !== null) {
            const main = document.getElementsByTagName("main")[0]
            // Using JS for layout is necessary because cannot be
            // laid out via CSS.
            let height = main.clientHeight - 20
            let width = main.clientWidth - 5
            plot(node, height, width, props)
        }
    }, [props]);


    return (
        <div
            ref={canvas}
        />
    )
}
export default Canvas
