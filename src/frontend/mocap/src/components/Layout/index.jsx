import { useContext, useState, useEffect } from "react";
import { Outlet, Link, useLocation } from "react-router-dom";

import 'bootstrap/dist/css/bootstrap.min.css';
//import * as Icon from 'react-bootstrap-icons';
import '../../App.css'; // Import the CSS file

function Layout(props){

    return <>
    <section className="nav">
        <nav className="navbar navbar-expand-lg bg-body-tertiary container-fluid">
          <div className="container-fluid">
            <a className="navbar-brand" style={{ color: 'white' }} href="#">
              MoCap
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
            <div className="collapse navbar-collapse" id="navbarSupportedContent">
              <ul className="navbar-nav me-auto mb-2 mb-lg-0"></ul>
              <li className="nav-item mx-2">
          <Link
            to="/"
            className="nav-link"
            activeClassName="active-link" // This class will be applied when the link is active
            style={{  color: 'white' }}
          >
            Home
          </Link>
        </li>
        <li className="nav-item mx-2">
          <Link
            to="/comp/"
            className="nav-link"
            activeClassName="active-link" // Apply the class for the active link
            style={{  color: 'white' }}
          >
            Analyze
          </Link>
        </li>
        <li className="nav-item mx-2">
          <Link
            to="/api/user/"
            className="nav-link"
            activeClassName="active-link" // Apply the class for the active link
            style={{  color: 'white' }}
          >
            Login
          </Link>
        </li>


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
          <i className="bi bi-facebook"></i>
        </a>
      </div>
    </section>

    <section className="text-white">
      <div className="container text-center text-md-start mt-5">

        <div className="row mt-3">
          <div className="col-md-3 col-lg-4 col-xl-3 mx-auto mb-4">

            <h6 className="text-uppercase fw-bold mb-4">
              <i className="fas fa-gem me-3"></i>Petpal Incorporated
            </h6>
            <p style={{marginTop: "20px",}}>
              Our aim is to connect loving families with their perfect furry companions. We strive to promote
              responsible pet ownership and reduce the number of animals in need by facilitating successful adoptions,
              creating forever homes, and supporting animal welfare organizations.
            </p>
          </div>
          <div className="col-md-4 col-lg-3 col-xl-3 mx-auto mb-md-0 mb-4">

            <h6 className="text-uppercase fw-bold mb-4" style={{marginLeft: "16px"}}>Contact</h6>
            <p><i className="bi bi-house me-3"></i> 100 Petpal St, Toronto, Ontario</p>
            <p>
              <i className="bi bi-envelope me-3"></i>
              petpal@gmail.com
            </p>
            <p><i className="bi bi-phone me-3"></i> + 01 234 567 88</p>
            <p><i className="bi bi-printer me-3"></i> + 01 234 567 89</p>
          </div>
        </div>
      </div>
    </section>
    <div className="text-center p-4 text-white" style={{backgroundColor: "rgba(0, 0, 0, 0.05)"}}>
      © 2021 Copyright Petpal
    </div>
  </footer>

    </>;
}

export default Layout;