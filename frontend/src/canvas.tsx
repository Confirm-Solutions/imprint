import React from 'react';
import Plotly from 'plotly.js-dist';
import { PlotType, MatrixData, getPlotlyData } from './data'


export interface CanvasState {
    plotType: PlotType
    checkboxStates: boolean[]
    data: MatrixData
    layerNames: string[]
}

async function plot(context: HTMLDivElement, height: number, width: number, state: CanvasState) {
    let method = Plotly.newPlot
    if ('data' in context) {
        console.log('updating plot in place')
        method = Plotly.react
    }

    var data = getPlotlyData(state.data, state.plotType, state.checkboxStates, state.layerNames)
    const layout: Partial<Plotly.Layout> = {
        height: height,
        width: width,
        autosize: false,
        legend: {
            font: {
                family: '"Roboto","Helvetica","Arial",sans-serif',
                size: 18
            },
        }

    }
    await method(context, data, layout);
}


class Canvas extends React.Component<CanvasState> {
    canvas: React.RefObject<HTMLDivElement>

    constructor(props: CanvasState) {
        super(props);
        this.canvas = React.createRef();
    }

    componentDidMount() {
        this.componentDidUpdate()
    }

    componentDidUpdate() {
        if (this.canvas.current) {
            const main = document.getElementsByTagName("main")[0]
            // Using JS for layout is necessary because canvas is not
            // laid out via CSS.
            let height = main.clientHeight - 20
            let width = main.clientWidth - 5
            console.log('plot dims', height, width)
            plot(this.canvas.current, height, width, this.props)
        }
    }

    render() {
        return (
            <div
                ref={this.canvas}
            />
        )
    }
}
export default Canvas
