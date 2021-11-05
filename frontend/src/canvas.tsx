import React from 'react';
import Plotly from 'plotly.js-dist';


interface CanvasState {
}

interface Coordinates3D {
    x: number[]
    y: number[]
    z: number[][]
}

// Fetch a static geometry for testing
async function getTestGeometry(): Promise<Coordinates3D> {
    let br = fetch('./B-55f3294ffc7ed8152742b504bc0001bdb0d0a0d8.csv')
    let pr = fetch('./P-55f3294ffc7ed8152742b504bc0001bdb0d0a0d8.csv')
    await Promise.all([br, pr])
    function parseCSV(text: string) {
        return text.trim().split('\n').map(l => l.split(',').map(s => parseFloat(s.trim())))
    }
    let bmat = parseCSV(await (await br).text())
    let pmat = parseCSV(await (await pr).text())

    // TODO: render the six type-1 error component as separate layers
    let bmat_sum = bmat.map(arr => arr[5]);
    let coords = bmat_sum.map((z, i) => [pmat[0][i], pmat[1][i], z])

    coords.sort()
    let x = coords.map(c => c[0])
    let y = coords.map(c => c[1])
    x = Array.from((new Set(x)).keys())
    y = Array.from((new Set(y)).keys())

    // NB: JS sorts alphabetically by default!
    x.sort((x, y) => x - y)
    y.sort((x, y) => x - y)
    let xi_map = new Map(x.map((x, i) => [x, i]))
    let yi_map = new Map(y.map((x, i) => [x, i]))

    // Plotly expects a 2D grid of z coordinates
    let z_mat: number[][] = []
    for (let i = 0; i < x.length; i++) {
        z_mat.push([])
        for (let j = 0; j < y.length; j++) {
            z_mat[z_mat.length - 1].push(NaN)
        }
    }
    coords.forEach(([x, y, z]) => {
        let xi = xi_map.get(x)
        let yi = yi_map.get(y)
        if (xi === undefined || yi === undefined) {
            return
        }
        z_mat[xi][yi] = z;
    })

    return { x: x, y: y, z: z_mat }
}

async function plot(context: HTMLDivElement, height: number, width: number) {

    let coords = await getTestGeometry()

    // Create a Plotly 3D surface plot
    // https://plotly.com/javascript/3d-surface-plots/
    // https://plotly.com/javascript/reference/surface/
    var data: Plotly.Data[] = [
        {
            type: 'surface',
            x: coords.x,
            y: coords.y,
            z: coords.z,
        }
    ];
    const layout: Partial<Plotly.Layout> = {
        height: height,
        width: width,
        autosize: false,

    }
    await Plotly.newPlot(context, data, layout);
}


class Canvas extends React.Component<CanvasState> {
    canvas: React.RefObject<HTMLDivElement>

    constructor(props: CanvasState) {
        super(props);
        this.canvas = React.createRef();
    }

    componentDidMount() {
        if (this.canvas.current) {
            const main = document.getElementsByTagName("main")[0]
            // Using JS for layout is necessary because canvas is not
            // laid out via CSS.
            let height = main.clientHeight - 20
            let width = main.clientWidth - 5
            console.log('plot dims', height, width)
            plot(this.canvas.current, height, width)
        }
        // this.componentDidUpdate()
    }

    componentDidUpdate() {
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