import React, { useState } from 'react';
import { Form, Button, Container, Row, Col, Card, Alert, Breadcrumb } from 'react-bootstrap';
import './style.css'; // Import the custom CSS file
import { Link } from 'react-router-dom';

const Register = () => {
const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [name, setName] = useState('');
    const [height, setHeight] = useState('');
    const [profilePicture, setProfilePicture] = useState(null);
        const [user, setUser] = useState(null);
    const [error, setError] = useState('');

    const handleRegister = (e) => {
        e.preventDefault();
        // Handle the registration logic here
        console.log({
            username,
            password,
            name,
            height,
            profilePicture
        });
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
                <Form.Label>Username</Form.Label>
                <Form.Control
                    type="text"
                    placeholder="Enter username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
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
