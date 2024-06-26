import React, { useState, useEffect } from 'react';
import ReactStars from 'react-rating-stars-component';
import './Reviews.css'; // Add your custom CSS

const Reviews = () => {
  //const [reviews, setReviews] = useState([]);
const reviews = [
  { name: 'John Doe', content: 'Great service and friendly staff!', rating: 5 },
  { name: 'Jane Smith', content: 'I loved the product quality.', rating: 4 },
  { name: 'Alice Johnson', content: 'Fast delivery and excellent customer support.', rating: 5 },
  { name: 'Bob Brown', content: 'Highly recommend this company to everyone.', rating: 4 },
  { name: 'Charlie Davis', content: 'Affordable prices and a wide range of products.', rating: 3 },
];

  const [newReview, setNewReview] = useState({ name: '', content: '', rating: 0 });

  useEffect(() => {
    fetchReviews();
  }, []);

  const fetchReviews = async () => {
    try {
      //const response = await axios.get('/api/reviews');
      //setReviews(response.data);
    } catch (error) {
      console.error('Error fetching reviews:', error);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewReview({ ...newReview, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      //await axios.post('/api/reviews', newReview);
      //setNewReview({ name: '', content: '' });
      fetchReviews();
    } catch (error) {
      console.error('Error submitting review:', error);
    }
  };

  return (
    <div className="reviews-section">
      <h2>Customer Reviews</h2>
      <ul className="reviews-list">
        {reviews.map((review, index) => (
          <li key={index} className="review-item">
            <h3>{review.name}</h3>
            <ReactStars
              count={5}
              value={review.rating}
              size={24}
              isHalf={true}
              edit={false}
              activeColor="#ffd700"
            />
            <p>{review.content}</p>
          </li>
        ))}
      </ul>
      {/*<form className="review-form" onSubmit={handleSubmit}>
        <input
          type="text"
          name="name"
          value={newReview.name}
          onChange={handleInputChange}
          placeholder="Your Name"
          required
        />
        <textarea
          name="content"
          value={newReview.content}
          onChange={handleInputChange}
          placeholder="Your Review"
          required
        ></textarea>
        <button type="submit">Submit Review</button>
      </form>*/}
    </div>
  );
};

export default Reviews;
