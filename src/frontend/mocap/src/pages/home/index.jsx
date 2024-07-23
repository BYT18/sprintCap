import React, { useEffect, useRef } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import './style.css'
import { Link } from 'react-router-dom';
import { motion, inView, Variants } from 'framer-motion';

import myImage1 from "../../assets/touchdown_R2.png";
import myImage2 from "../../assets/max_ver3.png";
import myImage3 from "../../assets/strike_R2.png";
import myImage4 from "../../assets/toe_off_L2.png";
import myImage5 from "../../assets/full_sup_L2.png";
import me from "../../assets/tang.png";
import bolt from "../../assets/usain.png";
import splitter from "../../assets/Content.png";

import Reviews from "../../components/Reviews/index.jsx";

const Home = () => {

 const newsPanelRef = useRef(null);
  const carouselRef = useRef(null);

  useEffect(() => {
    const adjustHeight = () => {
      if (newsPanelRef.current && carouselRef.current) {
        const carouselHeight = carouselRef.current.clientHeight;
        newsPanelRef.current.style.height = `${carouselHeight}px`;
      }
    };

    adjustHeight();
    window.addEventListener('resize', adjustHeight);
    return () => {
      window.removeEventListener('resize', adjustHeight);
    };
  }, []);

   const cardVariants: Variants = {
  offscreen: {
    y: 100,
    opacity:0
  },
  onscreen: {
    y: 0,
    //rotate: -10,
    transition: {
      type: "spring",
      bounce: 0.4,
      duration: 0.8,
    },
          opacity:1
  },
};

    const img1Variant: Variants = {
          offscreen: {
            x: -150
          },
          onscreen: {
            x: 0,
            //rotate: -100,
            transition: {
              type: "spring",
              bounce: 0.4,
              duration: 1.2
            }
          }
    };

    const img2Variant: Variants = {
          offscreen: {
            y: 150
          },
          onscreen: {
            y: 0,
            //rotate: -100,
            transition: {
              type: "spring",
              bounce: 0.4,
              duration: 1.2
            }
          }
    };

    const img3Variant: Variants = {
          offscreen: {
            x: 150
          },
          onscreen: {
            x: 0,
            //rotate: -100,
            transition: {
              type: "spring",
              bounce: 0.4,
              duration: 1.2
            }
          }
    };

  const listVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { staggerChildren: 0.1 } },
  };

  return (
  <body >
  <section class="banner text-sm-start text-center p-4">
    <div class="container">
      <div class="d-sm-flex justify-content-center align-items-center ">
        <div>
          <h1>A New Vision</h1>
          <p>At <span style={{font:'bold'}}>MoCap</span> we believe that technology should empower athletes to push beyond limits, break barriers, and achieve greatness. We envision a future where AI becomes an indispensable ally in an athlete's journey, offering real-time analysis and strategic guidance to enhance performance across various sports disciplines.</p>
        </div>
        <img class="img-fluid" src="https://png.pngtree.com/png-vector/20231115/ourmid/pngtree-young-man-sprinter-runner-running-silhouette-studio-shot-png-image_10589362.png" />
      </div>
    </div>
  </section>

  <div class="b-example-divider"></div>

  {/* Cards Section */}
  <motion.section className="team-section py-5 text-center" style={{display: "flex",flexDirection: "column",
      alignItems: "center", minHeight:"50vh"}}
                  variants={cardVariants}
            initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.5 }}
      >
      <div className="container">
    <div className="row">
        <div className="col-md-4 mb-4 mx-auto d-flex">
          <motion.div
            className="card homecards h-100 shadow-sm"
          >
            <div className="card-body d-flex flex-column justify-content-center">
              <h2 className="card-title">Innovation at the Core</h2>
              <p className="card-text">We thrive on pushing the boundaries of innovation. Our team of experts constantly refines our AI algorithms to stay ahead of the curve, ensuring you benefit from the latest advancements in sports technology.</p>
            </div>
          </motion.div>
        </div>
        <div className="col-md-4 mb-4 mx-auto d-flex">
          <motion.div
            className="card homecards h-100 shadow-sm"
            whileHover="hover"
            variants={cardVariants}
          >
            <div className="card-body d-flex flex-column justify-content-center">
              <h2 className="card-title">Data-Driven Excellence</h2>
              <p className="card-text">Harness the power of data to make informed decisions about your training and performance. MoCap transforms raw data into actionable intelligence, unlocking new levels of achievement.</p>
            </div>
          </motion.div>
        </div>
        <div className="col-md-4 mb-4 mx-auto d-flex">
          <motion.div
            className="card homecards h-100 shadow-sm"
            whileHover="hover"
            variants={cardVariants}
          >
            <div className="card-body d-flex flex-column justify-content-center">
              <h2 className="card-title">Personalized Experience</h2>
              <p className="card-text">We recognize that every athlete is unique. Our AI models adapt to your individual strengths, weaknesses, and goals, delivering a personalized experience that maximizes your athletic potential.</p>
            </div>
          </motion.div>
        </div>
            </div>
  </div>
      </motion.section>

{/* Animation Section */}
      <motion.section className="bike-fit-container"
           variants={cardVariants}
            initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.5 }}
      >
      <div className="bike-fit-images">
        <motion.img
            variants={img1Variant}
            initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.5 }}
        src={myImage4} alt="Biker 1" className="bike-fit-image" />

        <motion.img
        variants={img2Variant}
            initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.5 }}
        src={myImage2} alt="Biker 2" className="bike-fit-image" />

        <motion.img
        variants={img3Variant}
            initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.5 }}
        src={myImage5} alt="Biker 3" className="bike-fit-image" />
      </div>
      <motion.div
                   initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      className="bike-fit-info">
        <h2>#SPRINTFREE</h2>
        <h1>Get personalized coaching from anywhere, anytime</h1>
        <p>
Elevate your sprinting and running performance with our cutting-edge platform that combines industry-leading techniques with powerful A.I. Get personalized recommendations on your form and technique, consistent with top professional trainers. Optimize your running style anytime, anywhere, right from your phone. No appointments, no travel, just better performance.
        </p>
        <div className="bike-fit-buttons">
          <button className="get-started-button">Get Started</button>
        </div>
      </motion.div>
      </motion.section>

    <div className="image-section">
                <img src={splitter} alt="Background" className="background-image" />
                <div className="overlay-container">
                     <img
                        src={bolt} className="overlay-image" />
                     <img
                        src={me} className="overlay-image2" />
                 </div>
                    <div className="text-overlay">
                        <h1>Compare</h1>
                        <p>with the very best</p>
                    </div>
                </div>


      {/* News Section
      <motion.section className="news-section py-5" style={{ minHeight: "80vh",display: "flex",
      alignItems: "center"}}
           variants={cardVariants}
            initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.5 }}
      >
        <div className="container">
          <h2 className="text-center mb-4">Latest News</h2>
          <div className="row">
            <motion.div className="col-lg-4 mb-4"
            whileHover={{ scale: 1.05 }}
            >
              <motion.div
                className="news-panel p-3 border rounded shadow-sm"
                ref={newsPanelRef}
                initial="hidden"
                animate="visible"
                variants={listVariants}
              >
                <h3>Featured Stories</h3>
                <motion.ul className="list-unstyled">
                  {["AI Coaching Revolutionizes Training", "AI Predicts Game Outcomes", "Injury Prevention with AI", "AI Refereeing in Tennis", "Virtual Reality Training with AI"].map((story, index) => (
                    <motion.li key={index} variants={listVariants}
                     whileHover={{ scale: 1.025 }}
                    >
                      <a href="#">{story}</a>
                    </motion.li>
                  ))}
                </motion.ul>
              </motion.div>
            </motion.div>
            <motion.div className="col-lg-8"
            whileHover={{ scale: 1.05 }}
            >
              <div id="newsCarousel" className="carousel slide" data-ride="carousel" ref={carouselRef}>
                <div className="carousel-inner">
                  <div className="carousel-item active">
                    <img className="d-block w-100" src="https://dxbhsrqyrr690.cloudfront.net/sidearm.nextgen.sites/varsityblues.ca/images/2024/4/29/Brandon_Tang_Aru__2_.jpg" alt="News 1" />
                    <div className="carousel-caption pb-5">
                      <h5>AI Coaching Revolutionizes Training</h5>
                      <p>A top-tier football club adopts AI coaching assistants to analyze player performance during training sessions.</p>
                    </div>
                  </div>
                  <div className="carousel-item">
                    <img className="d-block w-100" src="https://img.olympics.com/images/image/private/t_s_pog_staticContent_hero_lg/f_auto/primary/hiuf5ahd3cbhr11q6m5m" alt="News 2" />
                    <div className="carousel-caption">
                      <h5>News 2</h5>
                      <p>Description of News 2</p>
                    </div>
                  </div>
                  <div className="carousel-item">
                    <img className="d-block w-100" src="https://builtforathletes.com/cdn/shop/articles/Usain_Bolt.jpg?v=1615809551" alt="News 3" />
                    <div className="carousel-caption">
                      <h5>News 3</h5>
                      <p>Description of News 3</p>
                    </div>
                  </div>
                </div>
                <a className="carousel-control-prev" href="#newsCarousel" role="button" data-slide="prev">
                  <span className="carousel-control-prev-icon" aria-hidden="true"></span>
                  <span className="sr-only">Previous</span>
                </a>
                <a className="carousel-control-next" href="#newsCarousel" role="button" data-slide="next">
                  <span className="carousel-control-next-icon" aria-hidden="true"></span>
                  <span className="sr-only">Next</span>
                </a>
              </div>
            </motion.div>
          </div>
        </div>
      </motion.section>
      */}

      <motion.section className="news-section py-5" style={{ minHeight: "80vh",display: "flex",alignItems: "center"}}
           variants={cardVariants}
            initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.5 }}
      >
        <div className="container">
            <Reviews />
        </div>
      </motion.section>

      {/* Team Section */}
      <motion.section className="team-section py-5" style={{display: "flex",
      alignItems: "center"}}
           variants={cardVariants}
            initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.5 }}
      >
        <div className="container">
          <h2 className="text-center mb-4">Our Team</h2>
          <div className="row text-center">
            <div className="col-md-4 mb-4">
              <motion.div
                className="team-member p-3 border rounded shadow-sm"
                whileHover={{ scale: 1.05, boxShadow: "0 8px 20px rgba(0, 0, 0, 0.15)" }}
                style={{ minHeight: "50vh",alignItems: "center", display: "flex", flexDirection: "column"}}
              >
                <h3>John Doe</h3>
                 <img class="img-fluid team-image" src="https://t4.ftcdn.net/jpg/02/19/63/31/360_F_219633151_BW6TD8D1EA9OqZu4JgdmeJGg4JBaiAHj.jpg"
                 style={{ maxHeight: "200px", objectFit: "cover", marginBottom: "1rem" }}
                 />
                <p>Lead Developer</p>
                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
              </motion.div>
            </div>
            <div className="col-md-4 mb-4">
              <motion.div
                className="team-member p-3 border rounded shadow-sm"
                whileHover={{ scale: 1.05, boxShadow: "0 8px 20px rgba(0, 0, 0, 0.15)" }}
                style={{display: "flex", flexDirection: "column", minHeight: "50vh",alignItems: "center"}}
              >
                <h3>Jane Smith</h3>
                 <img class="img-fluid team-image" src=" https://media.istockphoto.com/id/1317784594/photo/headshot-of-mature-50-years-old-asian-business-woman-on-grey-background.jpg?s=612x612&w=0&k=20&c=eOmdf5BbEG75m9MBSTvhjA5uMDmUj0zDtXd3lv0nm8U=
                "
                style={{ maxHeight: "200px", objectFit: "cover", marginBottom: "1rem" }}
                />
                <p>Project Manager</p>
                <p>Vivamus luctus urna sed urna ultricies ac tempor dui sagittis.</p>
              </motion.div>
            </div>
            <div className="col-md-4 mb-4">
              <motion.div
                className="team-member p-3 border rounded shadow-sm"
                whileHover={{ scale: 1.05, boxShadow: "0 8px 20px rgba(0, 0, 0, 0.15)" }}
                style={{ display: "flex", flexDirection: "column", minHeight: "50vh",alignItems: "center"}}
              >
                <h3>Bob Johnson</h3>
                 <img class="img-fluid team-image" src="https://media.istockphoto.com/id/1207856385/photo/joyful-happy-african-american-young-man-in-eyeglasses-portrait.jpg?s=612x612&w=0&k=20&c=M5sUFPE5xlF1fMxvNYgAqdpSZYKxSor3-SlF-o6IiJ0=
              "
              style={{ maxHeight: "200px", objectFit: "cover", marginBottom: "1rem" }}
              />
                  <p>UI/UX Designer</p>
                <p>Praesent dapibus, neque id cursus faucibus, tortor neque egestas auguae.</p>
              </motion.div>
            </div>
          </div>
        </div>
      </motion.section>

      {/* Call to Action Section */}
      <motion.section className="testimonials-section py-5" style={{ minHeight: "40vh",alignItems: "center"}}
           variants={cardVariants}
            initial="offscreen"
            whileInView="onscreen"
            viewport={{ once: true, amount: 0.5 }}
      >
        <div className="container text-center">
          <h2 className="mb-4">What Our Clients Say</h2>
          <p style={{color:'black'}}className="lead mb-4">"MoCap has transformed the way we train and perform. The AI-driven insights are a game-changer."</p>
           <Link to="/about/">
          <button className="btn btn-lg btn-light">Learn More</button>
             </Link>
        </div>
         </motion.section>
  </body>
  );
};

export default Home;
