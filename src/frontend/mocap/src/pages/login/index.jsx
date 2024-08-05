import React, { useState } from 'react';
import { Form, Button, Container, Row, Col, Card, Alert, Breadcrumb } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import './style.css'; // Import the custom CSS file

const Login = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [user, setUser] = useState(null);
    const [token, setToken] = useState('');
    const [name, setName] = useState('');
    const [error, setError] = useState('');

    const login = async (u, p) => {
        const formData = new FormData();
        formData.append("username", u);
        formData.append('password', p);
        try {
            //const response = await fetch('http://127.0.0.1:8000/login/', {
            const response = await fetch('http://3.143.116.75:8000/login/', {
                method: 'POST',
                headers: {},
                body: formData,
            });
            if (response.ok) {
                const data = await response.json();
                localStorage.clear();
                 localStorage.setItem('access_token', data['access']);
                 localStorage.setItem('refresh_token', data['refresh']);
                setToken(data['access']);
                await fetch_profile(data['access']);
            } else {
                setError('Invalid username or password');
                localStorage.clear();
            }
        } catch (error) {
            setError('Network error');
        }
    };

    const fetch_profile = async (accessToken) => {
        try {
            const response = await fetch('http://127.0.0.1:8000/prof/', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${accessToken}`,
                },
            });
            if (response.ok) {
                const data = await response.json();
                setName(data['name']);
                setUser(data);
                window.location.href = '/'
            } else {
                setError('Failed to fetch profile');
            }
        } catch (error) {
            setError('Network error');
        }
    };

    const handleLogin = async () => {
        setError('');
        await login(username, password);
    };

    const register = () => {
        window.location.href = '/'
    };

    return (
  <div className="container full-height">
            <Row className="justify-content-center w-100">
                <Col md={6} lg={4} md={8} lg={6} className="d-flex justify-content-center">
                    <Card className="shadow-lg p-4 min-width-card">
                        <Card.Body>
                            <h2 className="text-center mb-4">Login</h2>
                            {error && <Alert variant="danger">{error}</Alert>}
                            <Form>
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

                                <div className="button-container mt-3">
                                    <Button variant="primary" onClick={handleLogin} block>
                                      Login
                                     </Button>
                                    <Link className="text-button" to="/api/register/">Register here</Link>
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

export default Login;
