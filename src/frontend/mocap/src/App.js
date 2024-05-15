import logo from './logo.svg';
import './App.css';
import { BrowserRouter as Router, Route, Link, Routes } from 'react-router-dom';

import VidComp from './pages/upload/index'
import Layout from './components/Layout/index'
import Home from './pages/home/index'
import Login from './pages/login/index'

function App() {
  return (
    <main>
      <Router>
          <Routes>
            <Route path="/" element={<Layout />}>
                <Route index element={<Home/>} />
                <Route path='/comp/'  element={<VidComp/>} />



            </Route>
            <Route path='/api/user/'  element={<Login/>} />

          </Routes>
      </Router>
    </main>
  );
}

export default App;
