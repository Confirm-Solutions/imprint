var memoize = require("memoizee");

export type PlotType = "surface" | "scatter3d"
type Matrix2D = number[][]
type Coordinate = [number, number, number]
export interface MatrixData {
    pmat: Matrix2D,
    bmat: Matrix2D
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


// Fetch static matrices for testing
export const getTestMatrices: () => Promise<MatrixData> = memoize(async function () {
    let br = fetch('./B-55f3294ffc7ed8152742b504bc0001bdb0d0a0d8.csv')
    let pr = fetch('./P-55f3294ffc7ed8152742b504bc0001bdb0d0a0d8.csv')
    await Promise.all([br, pr])
    let data: MatrixData = { bmat: parseCSV(await (await br).text()), pmat: parseCSV(await (await pr).text()) }
    return data
})

export function parseCSV(text: string) {
    return text.trim().split('\n').map(l => l.split(',').map(s => parseFloat(s.trim())))
}


function getGeometry(data: MatrixData, type: PlotType, layer_index: number): SurfaceCoordinates3D | ScatterCoordinates3D {
    const bmat = data.bmat
    const pmat = data.pmat

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

export function getPlotlyData(matrices: MatrixData, plotType: PlotType, checkboxStates: boolean[], layerNames: string[]): Plotly.Data[] {

    // Create a Plotly 3D surface plot
    // https://plotly.com/javascript/3d-surface-plots/
    // https://plotly.com/javascript/reference/surface/
    var data: any[] = []

    for (let i = 0; i < checkboxStates.length; i++) {
        if (!checkboxStates[i]) {
            // Don't render this layer if the checkbox isn't checked
            continue
        }
        const coords = getGeometry(matrices, plotType, i);
        data.push(
            {
                name: layerNames[i],
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
