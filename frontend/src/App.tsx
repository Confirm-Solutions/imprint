import Canvas from './canvas';
import './App.css';


function App() {
  return (
    <div className="row">
      <div className="column left"></div>
      <div className="column middle">
        <article>
          <header>
            <h1>Simulated Family-Wise Error Rates</h1>
          </header>
          <main>
            <Canvas></Canvas>
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
