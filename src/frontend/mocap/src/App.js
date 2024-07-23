import logo from './logo.svg';
import './App.css';
//import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import { HashRouter as Router, Route, Routes } from 'react-router-dom';

import VidComp from './pages/upload/index'
import Layout from './components/Layout/index'
import Home from './pages/home/index'
import Login from './pages/login/index'
import Analysis from './pages/analysis/index'
import Kin from './pages/kin/index'
import Profile from './pages/profile/index'
import ScrollToTop from './components/ScrollToTop/index';
import Register from './pages/register/index'

function App() {
  return (
    <main>
      <Router>
      <ScrollToTop />
          <Routes>
            <Route path="/" element={<Layout />}>
                <Route index element={<Home/>} />
                <Route path='/comp/'  element={<VidComp/>} />
                <Route path='/analysis/'  element={<Analysis/>} />
                <Route path='/about/'  element={<Kin/>} />
                <Route path='/profile/'  element={<Profile/>} />
            </Route>
            <Route path='/api/user/'  element={<Login/>} />
            <Route path='/api/register/'  element={<Register/>} />
          </Routes>
      </Router>
    </main>
  );
}

export default App;
