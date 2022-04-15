import React, { ReactElement } from 'react';
import { Slider } from '@mui/material';

export interface Props {
    name: ReactElement;
    range: number[];
    setRange: React.Dispatch<React.SetStateAction<number[]>>;
    min: number;
    max: number;
}

function ParamSlider(props: Props) {
    return (<div className='sliderContainer'>
        <div className='paramName'>{props.name}</div>
        <div className='paramVal'>{props.range[0].toFixed(2)}</div>
        <div className='slider'>
            <Slider
                onChange={(e: any) => { props.setRange(e.target.value) }}
                value={props.range}
                step={1e-6}
                min={props.min}
                max={props.max}
            />
        </div>
        <div className='paramVal'>{props.range[1].toFixed(2)}</div>
    </div>)

}

export default ParamSlider;
