import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import './style.css'

const Home = () => {
  return (<body>
  <section class="banner text-sm-start text-center p-4">
    <div class="container">
      <div class="d-sm-flex justify-content-center align-items-center">
        <div>
          <h1>A New Vision</h1>
          <p>At <span style={{font:'bold'}}>MoCap</span> we believe that technology should empower athletes to push beyond limits, break barriers, and achieve greatness. [Your Sports Tech] envisions a future where AI becomes an indispensable ally in an athlete's journey, offering real-time analysis and strategic guidance to enhance performance across various sports disciplines.</p>
        </div>
        <img class="img-fluid" src="https://png.pngtree.com/png-vector/20231115/ourmid/pngtree-young-man-sprinter-runner-running-silhouette-studio-shot-png-image_10589362.png" />
      </div>
    </div>
  </section>

    <div class="b-example-divider"></div>

  <section class="container p-5">
    <div class="row">
      <div class="col-md-4 mb-4">
        <div class="homecards card h-100 text-center">
          <div class="card-body">
            <h2 class="card-title">Innovation at the Core</h2>
            <p class="card-text">We thrive on pushing the boundaries of innovation. Our team of experts constantly refines our AI algorithms to stay ahead of the curve, ensuring you benefit from the latest advancements in sports technology.</p>
          </div>
        </div>
      </div>
      <div class="col-md-4 mb-4">
        <div class="card h-100 text-center homecards">
          <div class="card-body">
            <h2 class="card-title">Data-Driven Excellence</h2>
            <p class="card-text">Harness the power of data to make informed decisions about your training and performance. MoCap transforms raw data into actionable intelligence, unlocking new levels of achievement.</p>
          </div>
        </div>
      </div>
      <div class="col-md-4 mb-4">
        <div class="card h-100 text-center homecards">
          <div class="card-body">
            <h2 class="card-title">Personalized Experience</h2>
            <p class="card-text">We recognize that every athlete is unique. Our AI models adapt to your individual strengths, weaknesses, and goals, delivering a personalized experience that maximizes your athletic potential.</p>
          </div>
        </div>
      </div>
       <div class="get-started-button d-flex justify-content-center align-items-center" style={{color: "white"}}>
            <p class="mb-0">Get Started Today!</p>
       </div>
    </div>
  </section>
  </body>
  );
};

export default Home;
