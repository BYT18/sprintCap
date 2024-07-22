import React from "react";
import { useContext, useState, useEffect } from "react";
import { Outlet, Link, useLocation, NavLink } from "react-router-dom";

import 'bootstrap/dist/css/bootstrap.min.css';
import "bootstrap-icons/font/bootstrap-icons.css";
import 'bootstrap/dist/js/bootstrap.bundle.min';

import * as Icon from 'react-bootstrap-icons';
import '../../App.css'; // Import the CSS file
import logo from '../../logo_transparent.png';

function Layout(props){
const location = useLocation();

  const [isAuth, setIsAuth] = useState(false);

  const getNavbarStyle = () => {
    console.log(location.pathname)
    switch (location.pathname) {
      case '/about/':
        return 'navbar-kin';
      default:
        return 'navbar-default';
    }
  };

  const getNavStyle = () => {
    console.log(location.pathname)
    switch (location.pathname) {
      case '/about/':
        return 'nav-kin';
      default:
        return 'nav';
    }
  };

     useEffect(() => {
     if (localStorage.getItem('access_token') !== null) {
        setIsAuth(true);
     }
    }, [isAuth]);

    return <>
    <section className={`nav ${getNavStyle()}`}>
        {/*<nav className="navbar navbar-expand-lg bg-body-tertiary container-fluid">*/}
        <nav className={`navbar ${getNavbarStyle()} navbar-expand-lg bg-body-tertiary container-fluid`}>
          <div className="container-fluid column">
            {/*<a className="navbar-brand" style={{ color: 'white' }} href="#">
              MoCap
            </a>*/}
            <a className="navbar-brand">
                <img src={logo}  alt="Logo" width="80" height="50" className="d-inline-block align-top" />
            </a>
            <button
              className="navbar-toggler bg-light"
              type="button"
              data-bs-toggle="collapse"
              data-bs-target="#navbarSupportedContent"
              aria-controls="navbarSupportedContent"
              aria-expanded="false"
              aria-label="Toggle navigation"
            >
              <span className="navbar-toggler-icon bg-light"></span>
            </button>
            <div className="collapse navbar-collapse justify-content-center" id="navbarSupportedContent">
              <ul className="navbar-nav me-auto mb-2 mb-lg-0 "></ul>
              <li className="nav-item mx-2 ">
                  <NavLink
                    to="/"
                    className={({ isActive }) => (isActive ? "nav-link active-link" : "nav-link")}
                    >
                    Home
                   </NavLink>
              </li>
        {/*<li className="nav-item mx-2">
          <NavLink
            to="/comp/"
            className={({ isActive }) => (isActive ? "nav-link active-link" : "nav-link")}
          >
            Analyze
           </NavLink>
        </li>*/}

        <li className="nav-item mx-2">
          <NavLink
            to="/about/"
           className={({ isActive }) => (isActive ? "nav-link active-link" : "nav-link")}
          >
            About
           </NavLink>
        </li>
        {isAuth && (<div>
        <li className="nav-item mx-2">
          <NavLink
            to="/analysis/"
            className={({ isActive }) => (isActive ? "nav-link active-link" : "nav-link")}
          >
            Analyze
           </NavLink>
        </li>
          <li className="nav-item mx-2">
          <NavLink
            to="/profile/"
            className={({ isActive }) => (isActive ? "nav-link active-link" : "nav-link")}
            >
            Profile
           </NavLink>
            </li>
            </div>
        )}
        {!isAuth && (
        <li className="nav-item mx-2">
          <NavLink
            to="/api/user/"
            className={({ isActive }) => (isActive ? "nav-link active-link" : "nav-link")}
            >
            Login
           </NavLink>
        </li>
        )}
          </div>
          </div>
        </nav>
      </section>



      <section>
        <Outlet />
      </section>

    <footer className="text-center text-lg-start bg-light text-muted foot">

    <section className="d-flex justify-content-center justify-content-lg-between p-4 border-bottom" style={{color: "white"}}>

      <div className="me-5 d-none d-lg-block">
        <span>Get connected with us on social networks:</span>
      </div>
      <div>
        <a href="" className="me-4 text-reset">
          <i className="bi bi-instagram"></i>
        </a>
        <a href="" className="me-4 text-reset">
          <i className="bi bi-threads"></i>
        </a>
        <a href="" className="me-4 text-reset">
          <i className="bi bi-snapchat"></i>
        </a>
        <a href="" className="me-4 text-reset">
          <i className="bi bi-facebook"></i>
        </a>
        <a href="" className="me-4 text-reset">
          <i className="bi bi-twitter"></i>
        </a>
      </div>
    </section>

    <section className="text-white">
     <div className="container text-center text-md-start mt-5">
  <div className="row mt-3">
    <div className="col-md-3 col-lg-4 col-xl-3 mx-auto mb-4">
      <h6 className="text-uppercase fw-bold mb-4">
        <i className="fas fa-gem me-3"></i>MoCap Sports Analysis
      </h6>
      <p style={{marginTop: "20px"}}>
        MoCap is dedicated to revolutionizing sports analysis through cutting-edge technology. Our mission is to empower athletes, coaches, and teams with AI-driven insights and tools to optimize performance, prevent injuries, and achieve their goals.
      </p>
    </div>
    <div className="col-md-4 col-lg-3 col-xl-3 mx-auto mb-md-0 mb-4">
      <h6 className="text-uppercase fw-bold mb-4" style={{marginLeft: "16px"}}>Contact</h6>
      <p><i className="bi bi-house me-3"></i> 123 MoCap Ave, New York, NY</p>
      <p><i className="bi bi-envelope me-3"></i> info@mocap.com</p>
      <p><i className="bi bi-phone me-3"></i> +1 123 456 7890</p>
    </div>
  </div>
</div>
</section>
{/*<div className="text-center p-4 text-white" style={{backgroundColor: "rgba(0, 0, 0, 0.05)"}}>*/}
<div className="text-center p-4 text-white">
  © 2021 Copyright MoCap
</div>

  </footer>

    </>;
}

export default Layout;