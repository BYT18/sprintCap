import React, { useState } from 'react';
import { Form, Button, Container, Row, Col, Card, Alert, Breadcrumb } from 'react-bootstrap';
import './style.css'; // Import the custom CSS file
import { Link } from 'react-router-dom';

const Register = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [name, setName] = useState('');
    const [height, setHeight] = useState('');
    const [profilePicture, setProfilePicture] = useState(null);
    const [user, setUser] = useState(null);
    const [error, setError] = useState('');
    const [token, setToken] = useState('');

    {/*const handleRegister = (e) => {
        e.preventDefault();
        // Handle the registration logic here
        console.log({
            username,
            password,
            name,
            height,
            profilePicture
        });
    };*/}

    const handleRegister = async (e) => {
        e.preventDefault();
        const formData = new FormData();
        formData.append("email", email);
        formData.append('password', password);
        try {
            //const response = await fetch('http://127.0.0.1:8000/register/', {
            //const response = await fetch('http://3.131.119.69:8000/login/', {
            const response = await fetch('/register/', {
                method: 'POST',
                headers: {},
                body: formData,
            });
            if (response.ok) {
                const data = await response.json();
                setUser(data['user'])
                localStorage.clear();
                localStorage.setItem('access_token', data['access']);
                localStorage.setItem('refresh_token', data['refresh']);
                console.log(data)
                console.log(data['access'])
                setToken(data['access']);
                await handleSubmit()
                //window.location.href = '/'
            } else {
                setError('Invalid username or password');
                localStorage.clear();
            }
        } catch (error) {
            setError('Network error');
        }
    };

     const handleSubmit = async () => {
        const formData = new FormData();
        formData.append('dob', '2000-01-01'); // Ensure dob is a valid date format
        formData.append('height', height);
        formData.append('name', "joe");
        formData.append('femur_len', 12);

        if (profilePicture) {
            formData.append('profile_pic', profilePicture);
        }

        const token = localStorage.getItem('access_token');
        try {
            //const response = await fetch('http://127.0.0.1:8000/create/', {
            //const response = await fetch('http://3.131.119.69:8000/create/', {
            const response = await fetch('/create/', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
                body: formData,
            });
            if (response.ok) {
                const data = await response.json();
                //setProfile(data);
                //setMessage('Profile updated successfully');
            } else {
                console.error('Failed to update profile');
                //setMessage('Failed to update profile');
            }
        } catch (error) {
            console.error('Network error', error);
            //setMessage('Network error');
        }
    };



    const handleFileChange = (e) => {
        setProfilePicture(e.target.files[0]);
    };



    return (
  <div className="container full-height">
            <Row className="justify-content-center w-100">
                <Col md={6} lg={4} md={8} lg={6} className="d-flex justify-content-center">
                    <Card className="shadow-lg p-4 min-width-card">
                        <Card.Body>
                            <h2 className="text-center mb-4">Register</h2>
                            {error && <Alert variant="danger">{error}</Alert>}
                            <Form onSubmit={handleRegister}>
            <Form.Group controlId="formUsername">
                <Form.Label>Email</Form.Label>
                <Form.Control
                    type="text"
                    placeholder="Enter email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                />
            </Form.Group>
            <Form.Group controlId="formPassword" className="mt-3">
                <Form.Label>Password</Form.Label>
                <Form.Control
                    type="password"
                    placeholder="Enter password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                />
            </Form.Group>
            <Form.Group controlId="formName" className="mt-3">
                <Form.Label>Name</Form.Label>
                <Form.Control
                    type="text"
                    placeholder="Enter your name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                />
            </Form.Group>
            <Form.Group controlId="formHeight" className="mt-3">
                <Form.Label>Height (cm)</Form.Label>
                <Form.Control
                    type="number"
                    placeholder="Enter your height"
                    value={height}
                    onChange={(e) => setHeight(e.target.value)}
                />
            </Form.Group>
            <Form.Group controlId="formProfilePicture" className="mt-3">
                <Form.Label>Profile Picture</Form.Label>
                <Form.Control
                    type="file"
                    onChange={handleFileChange}
                />
            </Form.Group>
            <div className="button-container mt-3">
                <Button variant="primary" type="submit" block>
                    Register
                </Button>
                <Link className="text-button" to="/api/user/">Login</Link>
            </div>
        </Form>
                            {user && (
                                <Alert variant="success" className="mt-3">
                                    Welcome {name}!
                                </Alert>
                            )}
                        </Card.Body>
                    </Card>
                </Col>
            </Row>
        </div>
    );
};

export default Register;
