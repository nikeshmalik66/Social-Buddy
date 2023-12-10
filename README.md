# Social Buddy
 A web application for Sentiment Analysis and Hate Speech Detection using BERT Model

# AI Model for Sentiment Analysis and Hate Speech Detection with Flask Integration

## Introduction
This project develops an AI model capable of performing sentiment analysis and detecting hate speech in text data. Leveraging advanced natural language processing (NLP) techniques and a robust Flask web framework, the system aims to provide an efficient and accurate tool for monitoring social media content.

## Technologies Used
- **Python**: Primary programming language.
- **Flask**: Web framework for integrating frontend and AI model.
- **BERT (Bidirectional Encoder Representations from Transformers)**: Core NLP model for sentiment analysis and hate speech detection.
- **Natural Language Processing**: Techniques for text preprocessing and analysis.
- **HTML, CSS, JavaScript**

## System Architecture
The system's architecture comprises several key components:
- **Text Preprocessing**: Tokenization, stopword removal, stemming, and lemmatization.
- **Sentiment Analysis**: Using NLP techniques to assess text sentiment.
- **Hate Speech Detection**: Identifying and classifying hate speech content.
- **Flask Integration**: Bridging the AI model with a user-friendly web interface.

## Dataset Description
The model was trained and evaluated on datasets containing social media posts and comments, labeled for sentiment and hate speech. These datasets feature a diverse range of expressions, offering a comprehensive foundation for robust model training.

## Types of Models
This is a Multimodal NLP Project. It consist of 4 different models which are trained on different dataset according to teh usecase
**Sentiment Analysis Model**: Identifies the sentiment or emotional tone expressed in a piece of text.
**Hate Speech Detection Model**: Aims to identify text that contains offensive or harmful language.
**Sarcasm Detection Model**: Detects instances of sarcasm in text, a crucial aspect for understanding nuanced communication.
**Slang Detection Model**: Focuses on identifying and understanding slang language expressions commonly used in informal communication.

## Labels in Model

**Sentiment Analysis Model**
This model has total 7 labels which are as follows:
- "anger"
- "disgust"
- "fear"
- "joy"
- "neutral"
- "sadness"
- "surprise"

### Other models have only 2 labels "True or 1" and "False or 0" 

## Model Training and Evaluation
The BERT model was fine-tuned and evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

The model demonstrated high accuracy and precision, indicating its effectiveness in sentiment analysis and hate speech detection.

## Implementation Details
- Collect data from the mentioned Kaggle datasets, which contain tweets with hate speech and offensive language.
- Extracted relevant features from the cleaned and preprocessed text data.Split the dataset into training and testing sets to evaluate the performance of the chosen models
- Split the dataset into training and testing sets to evaluate the performance of the chosen models
- Train the chosen models on the training dataset, performing hyperparameter tuning if necessary.
- Web application created using HTML, CSS and JavaScript
- The Flask application serves as the interface for the AI model, enabling users to input text data for analysis. The integration focuses on real-time processing and display of analysis results.

## Demo Screenshot

![Demo Screenshot 1](static\project-screenshots\demo-1.jpg)
*Screenshot of the web application Home Page*

![Demo Screenshot 1](static\project-screenshots\demo-2.png)
*Screenshot of the web application interface with the AI model*

![Demo Screenshot 2](static\project-screenshots\demo-3.png)
*Screenshot of interaction with the AI model*

### Other models have similar interfaces

## Installation and Setup

run the app.py in the project folder

## Future Work and Enhancements
Future enhancements include improving the model's accuracy with larger datasets, implementing multilingual support, and refining the Flask interface for better user experience.

## Contributors
- Nikesh Jagdish Malik
- Akash Jayaprasad Nair
- Ayush Radheshyam Prajapati
