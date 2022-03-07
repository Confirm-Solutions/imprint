import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders canvas', () => {
  const { container } = render(<App />);
  let canvas = container.querySelector("canvas");
  expect(canvas).toBeDefined();
});
