import React, { useState, useEffect } from 'react';
import { Form, Button, Container, Row, Col, Card, Alert, Modal } from 'react-bootstrap';
import './style.css';
import {
  motion,
  useScroll,
  useSpring,
  useTransform,
  MotionValue
} from "framer-motion";

const Profile = () => {
    const [profile, setProfile] = useState(null);
    const [height, setHeight] = useState('');
    const [leg, setLeg] = useState('');
    const [name, setName] = useState('');
    const [profileImage, setProfileImage] = useState(null);
    const [message, setMessage] = useState('');

    const [image, setImage] = useState(null);
    const [imageURL, setImageURL] = useState('');

    const [show, setShow] = useState(false);
    const handleClose = () => {
        setShow(false)
        setShowLength(false)
    };
    const handleShow = () => setShow(true);
    const [showLength, setShowLength] = useState(false);

    const [slider, setSlider] = useState(30); // Default height value (in cm)

      const handleSliderChange = (event) => {
        setSlider(event.target.value);
      };


    useEffect(() => {
        const fetchProfile = async () => {
            const token = localStorage.getItem('access_token');
            try {
                //const response = await fetch('http://127.0.0.1:8000/prof/', {
                const response = await fetch('http://3.143.116.75:8000/prof/', {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${token}`,
                    },
                });
                if (response.ok) {
                    const data = await response.json();
                    setProfile(data);
                    setHeight(data.height || '');
                    setName(data.name || '');
                } else {
                    console.error('Failed to fetch profile');
                }
            } catch (error) {
                console.error('Network error', error);
            }
        };

        fetchProfile();
    }, []);

    const logout = () => {
       localStorage.clear();
       window.location.href = '/'
    };

    const handleImageChange = (e) => {
        setProfileImage(e.target.files[0]);
    };

    const handleImageUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            setImage(file);
            setImageURL(URL.createObjectURL(file));
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const formData = new FormData();
        formData.append('height', height);
        if (profileImage) {
            formData.append('profile_pic', profileImage);
        }

        const token = localStorage.getItem('access_token');
        try {
            const response = await fetch('http://127.0.0.1:8000/prof/', {
                method: 'PUT',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
                body: formData,
            });
            if (response.ok) {
                const data = await response.json();
                setProfile(data);
                setMessage('Profile updated successfully');
            } else {
                console.error('Failed to update profile');
                setMessage('Failed to update profile');
            }
        } catch (error) {
            console.error('Network error', error);
            setMessage('Network error');
        }
    };

    const urlToFile = async (url, filename) => {
        const response = await fetch(url);
        const blob = await response.blob();
        return new File([blob], filename, { type: blob.type });
    };

    const handleMeasure = async (e) => {
        e.preventDefault();
        const file = await urlToFile(imageURL, 'image.jpg'); // Convert URL to File object
        const formData = new FormData();
        formData.append('leg', slider);
        formData.append('img', file);
        console.log(imageURL)

        const token = localStorage.getItem('access_token');
        try {
            const response = await fetch('http://127.0.0.1:8000/prof/', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
                body: formData,
            });
            if (response.ok) {
                const data = await response.json();
                //setProfile(data);
                console.log(data)
                setLeg(data)
                setShowLength(true)
            } else {
                console.error('Failed to update profile');
                setMessage('Failed to update profile');
            }
        } catch (error) {
            console.error('Network error', error);
            setMessage('Network error');
        }
    };

    //if (!profile) return <div>Loading...</div>;

    return (
      <section className="bannerCards text-sm-start text-center p-4">
            <motion.div className="container"initial={{ opacity: 0, scale: 0.5 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{
        duration: 0.8,
        delay: 0.5,
        ease: [0, 0.71, 0.2, 1.01]
      }}>
                <Row className="justify-content-md-center">
                    <Col md={12}>
                        <Card className="shadow-lg p-4 card-custom-width">
                            <Card.Body>
                                {message && <Alert variant="info">{message}</Alert>}
                                <Form onSubmit={handleSubmit}>
                                    <Form.Group controlId="formName">
                                        <Form.Control
                                            type="text"
                                            placeholder="Enter name"
                                            value={name}
                                            onChange={(e) => setName(e.target.value)}
                                            className="text-box-as-h2"
                                        />
                                         <img
                                        alt="https://img.freepik.com/premium-vector/man-avatar-profile-picture-vector-illustration_268834-538.jpg"
                                        src="https://varsityblues.ca/images/2024/1/5/Tang_Brandon.jpg?width=300"
                                        className="circular-image mt-3"
                                    />
                                    </Form.Group>
                                    <Form.Group controlId="formHeight">
                                        <Form.Label>Height (cm)</Form.Label>
                                        <Form.Control
                                            type="number"
                                            placeholder="Enter height"
                                            value={height}
                                            onChange={(e) => setHeight(e.target.value)}
                                        />
                                    </Form.Group>
                                    <Form.Group controlId="formProfileImage" className="mt-3">
                                        {/*<Button variant="primary" onClick={handleShow}>
                                            Launch Body Calibration
                                        </Button>*/}
                                        <div class="scan-button-container">
                                            <button class="round-button" onClick={handleShow}>Launch Calibration</button>
                                        </div>
                                    </Form.Group>
                                    <div className="button-container mt-3">
                                    <Button variant="primary" type="submit" block className="mt-5">
                                        Save Profile
                                    </Button>
                                    <Button variant="secondary" type="submit" block className="mt-5" onClick={logout}>
                                        Logout
                                    </Button>
                                </div>
                                </Form>
                            </Card.Body>
                        </Card>
                    </Col>
                </Row>
                 <Modal
      size="lg"
      aria-labelledby="contained-modal-title-vcenter"
      centered
      show={show} onHide={handleClose}
    >
      <Modal.Header closeButton>
        <Modal.Title id="contained-modal-title-vcenter">
          Scan
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <h4>Dimension</h4>
        <p>
          Please load a picture of yourself facing forward and give inseam length of pants to help with camera calibration when analyzing video.
        </p>
        <div className="container mt-2">
              <div className="slider-container">
              <label style={{color:"black"}} htmlFor="height-slider">Select Pant Length: {slider} inches</label>
              <input
                type="range"
                id="height-slider"
                min="0"
                max="50"
                value={slider}
                onChange={handleSliderChange}
                className="slider"
              />
            </div>
        </div>
        <input class="form-control mb-2" type="file" id="formFileMultiple" accept="image/*" onChange={handleImageUpload}/>
          {imageURL && (
            <div className="image-container mt-3">
                <img src={imageURL} alt="Selected" className="img-fluid" />
            </div>
          )}
          {showLength && (
          <div className="image-container">
            <p>{leg}</p>
             </div>
          )
          }
      </Modal.Body>
      <Modal.Footer>
        <Button onClick={handleClose}>Close</Button>
        <Button onClick={handleMeasure}>Scan</Button>
      </Modal.Footer>
    </Modal>
            </motion.div>
        </section>
    );
};

export default Profile;
