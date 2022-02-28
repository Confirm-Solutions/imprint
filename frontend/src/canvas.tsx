import React from 'react';
import Plotly from 'plotly.js-dist';
var memoize = require("memoizee");


export type PlotType = "surface" | "scatter3d"
type Matrix2D = number[][]
type Coordinate = [number, number, number]

export interface CanvasState {
    plotType: PlotType
    checkboxStates: boolean[]
}

interface SurfaceCoordinates3D {
    x: number[]
    y: number[]
    z: number[][]
}

interface ScatterCoordinates3D {
    x: number[]
    y: number[]
    z: number[]
}


const getTestMatrices: () => Promise<[Matrix2D, Matrix2D]> = memoize(async function () {
    let br = fetch('./B-55f3294ffc7ed8152742b504bc0001bdb0d0a0d8.csv')
    let pr = fetch('./P-55f3294ffc7ed8152742b504bc0001bdb0d0a0d8.csv')
    await Promise.all([br, pr])
    function parseCSV(text: string) {
        return text.trim().split('\n').map(l => l.split(',').map(s => parseFloat(s.trim())))
    }
    let bmat = parseCSV(await (await br).text())
    let pmat = parseCSV(await (await pr).text())
    return [bmat, pmat]
})


// Fetch a static geometry for testing
async function getTestGeometry(type: PlotType, layer_index: number): Promise<SurfaceCoordinates3D | ScatterCoordinates3D> {
    const [bmat, pmat] = await getTestMatrices()

    // TODO: render the six type-1 error component as separate layers
    let bmat_sum = bmat.map(arr => arr[layer_index]);
    let coords: Coordinate[] = bmat_sum.map((z, i) => [pmat[0][i], pmat[1][i], z])

    if (type === 'scatter3d') {
        return {
            x: coords.map(a => a[0]),
            y: coords.map(a => a[1]),
            z: coords.map(a => a[2]),
        }
    }
    // surfact plot type
    function getSortedDeduplicatedCoords(coordIndex: number) {
        let x = coords.map(c => c[coordIndex])
        x = Array.from((new Set(x)).keys())

        // NB: JS sorts alphabetically by default!
        x.sort((x, y) => x - y)
        return x
    }
    const x = getSortedDeduplicatedCoords(0)
    const y = getSortedDeduplicatedCoords(1)
    const xi_map = new Map(x.map((x, i) => [x, i]))
    const yi_map = new Map(y.map((x, i) => [x, i]))

    // Plotly expects a 2D grid of z coordinates
    let z_mat: number[][] = new Array(x.length).fill(0).map(() => new Array(y.length).fill(NaN))
    coords.forEach(([x, y, z]) => {
        let xi = xi_map.get(x)
        let yi = yi_map.get(y)
        if (xi === undefined || yi === undefined) {
            return
        }
        // Convention is ys first
        z_mat[yi][xi] = z;
    })

    return { x: x, y: y, z: z_mat }
}

async function getData(plotType: PlotType, checkboxStates: boolean[]): Promise<Plotly.Data[]> {

    // Create a Plotly 3D surface plot
    // https://plotly.com/javascript/3d-surface-plots/
    // https://plotly.com/javascript/reference/surface/
    var data: any[] = []

    for (let i = 0; i < checkboxStates.length; i++) {
        if (!checkboxStates[i]) {
            // Don't render this layer if the checkbox isn't checked
            continue
        }
        const coords = await getTestGeometry(plotType, i);
        data.push(
            {
                type: plotType,
                mode: 'markers',
                marker: {
                    size: 6,
                    line: {
                        color: 'rgba(217, 217, 217, 0.14)',
                        width: 0.5
                    },
                    opacity: 0.8
                },
                coloraxis: "coloraxis",
                ...coords,
            }
        )
    }
    return data;
}


async function plot(context: HTMLDivElement, height: number, width: number, checkboxStates: boolean[], plotType: PlotType = 'surface') {
    let method = Plotly.newPlot
    if ('data' in context) {
        console.log('updating plot in place')
        method = Plotly.react
    }

    var data: Plotly.Data[] = await getData(plotType, checkboxStates);
    const layout: Partial<Plotly.Layout> = {
        height: height,
        width: width,
        autosize: false,

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
            plot(this.canvas.current, height, width, this.props.checkboxStates, this.props.plotType)
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
